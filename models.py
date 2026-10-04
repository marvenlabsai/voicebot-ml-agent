"""The STT, LLM and TTS a call runs on, built from what the backend sends with the call:

    "language": "hi",
    "stt": {"provider": "deepgram", "model": "nova-3", "language": "hi"},
    "llm": {"provider": "cerebras", "model": "gpt-oss-120b"},
    "tts": {"provider": "cartesia", "model": "sonic-3.6", "voiceId": "…", "language": "hi"},

Each provider gets its own language tag (the backend looks them up per language). Adding a
provider means adding a builder below and the option in backend/src/lib/catalog.js.
"""

from __future__ import annotations

import logging

from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("voice-agent.models")

DEFAULTS = {
    "stt": {"provider": "deepgram", "model": "nova-3"},
    "llm": {"provider": "cerebras", "model": "gpt-oss-120b"},
    "tts": {"provider": "cartesia", "model": "sonic-3.6"},
}

STT_BUILDERS = {
    "deepgram": lambda c: deepgram.STT(model=c["model"], language=c["language"]),
}
LLM_BUILDERS = {
    "cerebras": lambda c: openai.LLM.with_cerebras(model=c["model"]),
}
TTS_BUILDERS = {
    "cartesia": lambda c: cartesia.TTS(model=c["model"], language=c["language"], **({"voice": c["voiceId"]} if c.get("voiceId") else {})),
}
BUILDERS = {"stt": STT_BUILDERS, "llm": LLM_BUILDERS, "tts": TTS_BUILDERS}


def speech_config(config: dict) -> dict:
    """stt/llm/tts settings for this call; an unknown provider falls back to the default one."""
    language = config.get("language") or "en"
    out = {}
    for kind, default in DEFAULTS.items():
        c = {**default, **(config.get(kind) or {})}
        if c["provider"] not in BUILDERS[kind]:
            logger.warning("unknown %s provider %r, using %s", kind, c["provider"], default["provider"])
            c = {**default, **({"voiceId": c["voiceId"]} if kind == "tts" and c.get("voiceId") else {})}
        if kind != "llm":
            c.setdefault("language", language)
        out[kind] = c
    return out


def build(kind: str, c: dict):
    return BUILDERS[kind][c["provider"]](c)
