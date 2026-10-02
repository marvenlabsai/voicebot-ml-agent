"""Matches what the caller said against the script's scenarios with a small local embedding model.

Each scenario has example phrases. A turn matches the scenario whose closest example is most
similar to the caller's text, if that score clears the scenario's threshold *and* beats the
runner-up by MARGIN. Otherwise there's no match and the LLM answers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("voice-agent.script")

# Multilingual (English, Hindi, Hinglish), ~220 MB, ~5 ms per turn on CPU
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MODEL_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "models"))
# The best scenario must beat the next-best one by this much, so near-ties go to the LLM
MARGIN = float(os.getenv("SCRIPT_MATCH_MARGIN", "0.05"))
DEFAULT_THRESHOLD = 0.75
SLOW_EMBED_MS = 50

_model = None
_model_lock = threading.Lock()
# phrase -> normalized vector, shared by every call in this worker process
_vector_cache: dict[str, np.ndarray] = {}


def load_model():
    """Loads the embedding model once per process (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding

                start = time.perf_counter()
                _model = TextEmbedding(EMBED_MODEL, cache_dir=MODEL_CACHE_DIR)
                logger.info("loaded embedding model %s in %.1fs", EMBED_MODEL, time.perf_counter() - start)
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(load_model().embed(texts)), dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def _embed_cached(texts: list[str]) -> np.ndarray:
    missing = list({t for t in texts if t not in _vector_cache})
    if missing:
        for text, vector in zip(missing, _embed(missing)):
            _vector_cache[text] = vector
    return np.stack([_vector_cache[t] for t in texts])


@dataclass
class Scenario:
    id: str
    name: str
    reply: str
    next: str  # step id | "stay" | "end"
    threshold: float
    vectors: np.ndarray  # (n_examples, dim), normalized


@dataclass
class Match:
    scenario: Scenario
    score: float
    runner_up: float


class ScriptRouter:
    def __init__(self, script: dict):
        default_threshold = float(script.get("threshold") or DEFAULT_THRESHOLD)
        self._raw = script
        self._default_threshold = default_threshold
        self.start_step: str | None = (script.get("steps") or [{}])[0].get("id")
        self._steps: dict[str, list[Scenario]] = {}
        self._global: list[Scenario] = []

    def replies(self) -> list[str]:
        """Every scripted line, for pre-synthesis."""
        out = [sc["reply"] for st in self._raw.get("steps") or [] for sc in st.get("scenarios") or []]
        out += [sc["reply"] for sc in self._raw.get("global") or []]
        return list(dict.fromkeys(r for r in out if r))

    def prepare(self) -> None:
        """Embeds every example phrase (blocking; run it in a thread)."""

        def build(sc: dict) -> Scenario:
            examples = [e for e in sc.get("examples") or [] if e.strip()]
            return Scenario(
                id=sc["id"],
                name=sc.get("name") or sc["id"],
                reply=sc["reply"],
                next=sc.get("next") or "stay",
                threshold=float(sc.get("threshold") or self._default_threshold),
                vectors=_embed_cached(examples),
            )

        self._steps = {
            st["id"]: [build(sc) for sc in st.get("scenarios") or [] if sc.get("examples")]
            for st in self._raw.get("steps") or []
        }
        self._global = [build(sc) for sc in self._raw.get("global") or [] if sc.get("examples")]

    def match_sync(self, text: str, step_id: str | None) -> tuple[Match | None, float]:
        """Returns (match or None, best score seen)."""
        candidates = self._steps.get(step_id or "", []) + self._global
        if not candidates or not text.strip():
            return None, 0.0
        start = time.perf_counter()
        query = _embed([text])[0]
        took_ms = (time.perf_counter() - start) * 1000
        if took_ms > SLOW_EMBED_MS:
            logger.warning("slow embedding: %.0f ms", took_ms)

        scored = sorted(((float((sc.vectors @ query).max()), sc) for sc in candidates), key=lambda x: -x[0])
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= best.threshold and best_score - runner_up >= MARGIN:
            return Match(best, best_score, runner_up), best_score
        return None, best_score

    async def match(self, text: str, step_id: str | None) -> tuple[Match | None, float]:
        return await asyncio.to_thread(self.match_sync, text, step_id)
