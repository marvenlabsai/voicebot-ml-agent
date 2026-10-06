"""The models a call runs on, built from what the backend sends with the call.

Pipeline mode (speech-to-text, LLM and text-to-speech):

    "mode": "pipeline",
    "language": "hi",
    "stt": {"provider": "deepgram", "model": "nova-3", "language": "hi"},
    "llm": {"provider": "cerebras", "model": "gpt-oss-120b"},
    "tts": {"provider": "cartesia", "model": "sonic-3.6", "voiceId": "…", "language": "hi"},

Realtime mode (one speech-to-speech model that listens and speaks with its own voice):

    "mode": "realtime",
    "realtime": {"provider": "openai", "model": "gpt-live-1", "voiceId": "marin"},
    (or {"provider": "google", "model": "gemini-3.8-live", "voiceId": "Puck"})

Each provider gets its own language tag (the backend looks them up per language). Adding a
provider means adding a builder below and the option in backend/src/lib/catalog.js.
"""

from __future__ import annotations

import logging
import os

from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("voice-agent.models")

DEFAULTS = {
    "stt": {"provider": "deepgram", "model": "nova-3"},
    "llm": {"provider": "cerebras", "model": "gpt-oss-120b"},
    "tts": {"provider": "cartesia", "model": "sonic-3.6"},
    "realtime": {"provider": "openai", "model": "gpt-live-1", "voiceId": "marin"},
}

# GPT-Live hands reasoning and tool calls to a backend model; empty = OpenAI's default for it
GPT_LIVE_BACKEND_MODEL = os.getenv("GPT_LIVE_BACKEND_MODEL", "")

STT_BUILDERS = {
    "deepgram": lambda c: deepgram.STT(model=c["model"], language=c["language"]),
}
LLM_BUILDERS = {
    "cerebras": lambda c: openai.LLM.with_cerebras(model=c["model"]),
}
TTS_BUILDERS = {
    "cartesia": lambda c: cartesia.TTS(model=c["model"], language=c["language"], **({"voice": c["voiceId"]} if c.get("voiceId") else {})),
}
def _gemini_live(c: dict):
    """Gemini Live (e.g. gemini-3.8-live). It picks the spoken language itself (the prompt names
    it), and 3.8 rejects thinking settings and affective dialog, so neither is sent.

    Gemini Live's speech detection defaults to high start-of-speech sensitivity, so a cough or a
    short "hmm" on a phone line cuts the agent off; low sensitivity keeps real interruptions
    working without that."""
    from google.genai import types
    from livekit.plugins import google  # only loaded for agents that use it

    return google.realtime.RealtimeModel(
        model=c["model"],
        voice=c.get("voiceId") or "Puck",
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
            ),
        ),
    )


REALTIME_BUILDERS = {
    "openai": lambda c: openai.realtime.GPTLiveModel(
        model=c["model"],
        voice=c.get("voiceId") or "marin",
        **({"responses_options": {"model": GPT_LIVE_BACKEND_MODEL}} if GPT_LIVE_BACKEND_MODEL else {}),
    ),
    "google": _gemini_live,
}
BUILDERS = {"stt": STT_BUILDERS, "llm": LLM_BUILDERS, "tts": TTS_BUILDERS, "realtime": REALTIME_BUILDERS}


def is_realtime(config: dict) -> bool:
    return config.get("mode") == "realtime"


def speech_config(config: dict) -> dict:
    """stt/llm/tts (or realtime) settings for this call; an unknown provider falls back to the default."""
    language = config.get("language") or "en"
    out = {"mode": "realtime" if is_realtime(config) else "pipeline"}
    for kind, default in DEFAULTS.items():
        c = {**default, **(config.get(kind) or {})}
        if c["provider"] not in BUILDERS[kind]:
            logger.warning("unknown %s provider %r, using %s", kind, c["provider"], default["provider"])
            c = {**default, **({"voiceId": c["voiceId"]} if kind == "tts" and c.get("voiceId") else {})}
        if kind in ("stt", "tts"):
            c.setdefault("language", language)
        out[kind] = c
    return out


def build(kind: str, c: dict):
    return BUILDERS[kind][c["provider"]](c)


def realtime_call_config(config: dict) -> dict:
    """A realtime model speaks only through its own audio. The opening line is kept (it's asked
    for word for word, see opening_line_instructions); the end-call message, silence prompt and
    features built on the speech pipeline (filler-word filter, scripted replies) don't apply.
    The model is told the language."""
    if not is_realtime(config):
        return config
    language = config.get("languageName") or config.get("language") or "English"
    prompt = (config.get("prompt") or "").rstrip()
    out = {k: v for k, v in config.items() if k not in ("endCallMessage", "fillerWords", "script")}
    out["prompt"] = f"{prompt}\n\nSpeak with the caller in {language}."
    out["silence"] = {**(config.get("silence") or {}), "message": ""}
    return out


def opening_line_instructions(greeting: str) -> str:
    """A realtime model has no "say this text" call; it's asked to open with the line verbatim."""
    return (
        "Open the call now with the exact text below, word for word. Do not add, drop or change "
        "any words, and do not wait for the caller to speak first.\n\n" + greeting
    )
