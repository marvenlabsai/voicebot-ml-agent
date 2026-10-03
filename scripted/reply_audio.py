"""Pre-synthesizes scripted lines so they can play the instant a scenario matches.

Lines are synthesized with the call's own TTS (same voice and language) when the call starts,
which overlaps with ringing on phone calls. Each line is also written to a WAV on disk, keyed by
voice + language + text, so lines without per-call variables are only ever synthesized once.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import wave
from collections.abc import AsyncIterator
from pathlib import Path

from livekit import rtc
from livekit.agents import tts as lk_tts

logger = logging.getLogger("voice-agent.script")

CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "tts")))
# Played in small frames so an interruption can cut in quickly
FRAME_MS = 20
MAX_PARALLEL = 4


class ReplyAudio:
    def __init__(self, tts: lk_tts.TTS, cache_key: str):
        self._tts = tts
        self._cache_key = cache_key  # voice|language|model
        self._frames: dict[str, rtc.AudioFrame] = {}
        self._task: asyncio.Task | None = None
        # Cache stats for the call's usage report
        self.stats = {"linesFromCache": 0, "linesSynthesized": 0, "charactersSynthesized": 0, "linesPlayed": 0}

    def start(self, lines: list[str]) -> None:
        """Begins synthesizing in the background; play() uses whatever is ready."""
        self._task = asyncio.create_task(self._prepare(lines))

    async def aclose(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    def _path(self, text: str) -> Path:
        digest = hashlib.sha1(f"{self._cache_key}|{text}".encode()).hexdigest()
        return CACHE_DIR / f"{digest}.wav"

    async def _prepare(self, lines: list[str]) -> None:
        sem = asyncio.Semaphore(MAX_PARALLEL)

        async def one(text: str) -> None:
            async with sem:
                try:
                    frame = await asyncio.to_thread(self._read, self._path(text))
                    if frame is None:
                        async with self._tts.synthesize(text) as stream:
                            frame = await stream.collect()
                        await asyncio.to_thread(self._write, self._path(text), frame)
                        self.stats["linesSynthesized"] += 1
                        self.stats["charactersSynthesized"] += len(text)
                    else:
                        self.stats["linesFromCache"] += 1
                    self._frames[text] = frame
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Not fatal: say() will synthesize this line live instead
                    logger.warning("could not pre-synthesize %r", text[:60], exc_info=True)

        await asyncio.gather(*(one(t) for t in dict.fromkeys(lines) if t.strip()))
        logger.info("pre-synthesized %d/%d scripted lines", len(self._frames), len(set(lines)))

    @staticmethod
    def _read(path: Path) -> rtc.AudioFrame | None:
        if not path.exists():
            return None
        try:
            with wave.open(str(path), "rb") as f:
                data = f.readframes(f.getnframes())
                return rtc.AudioFrame(data, f.getframerate(), f.getnchannels(), f.getnframes())
        except Exception:
            logger.warning("bad TTS cache file %s, re-synthesizing", path.name)
            return None

    @staticmethod
    def _write(path: Path, frame: rtc.AudioFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with wave.open(str(tmp), "wb") as f:
            f.setnchannels(frame.num_channels)
            f.setsampwidth(2)  # int16
            f.setframerate(frame.sample_rate)
            f.writeframes(bytes(frame.data))
        os.replace(tmp, path)

    def play(self, text: str) -> AsyncIterator[rtc.AudioFrame] | None:
        """Frames for a pre-synthesized line, or None if it isn't ready (synthesize it live)."""
        frame = self._frames.get(text)
        if frame is None:
            return None
        self.stats["linesPlayed"] += 1

        async def frames() -> AsyncIterator[rtc.AudioFrame]:
            step = frame.sample_rate * FRAME_MS // 1000
            data = memoryview(frame.data).cast("B").cast("h")
            channels = frame.num_channels
            total = frame.samples_per_channel
            for start in range(0, total, step):
                n = min(step, total - start)
                chunk = data[start * channels : (start + n) * channels]
                yield rtc.AudioFrame(chunk.tobytes(), frame.sample_rate, channels, n)

        return frames()
