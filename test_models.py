"""STT / LLM / TTS built from the per-call config the backend sends."""

import asyncio

import models
from models import build, speech_config

CONFIG = {
    "language": "hi",
    "stt": {"provider": "deepgram", "model": "nova-3", "language": "hi"},
    "llm": {"provider": "cerebras", "model": "gpt-oss-120b"},
    "tts": {"provider": "cartesia", "model": "sonic-3.6", "voiceId": "4459a9a5-69d6-4680-b970-e13dc51845b6", "language": "hi"},
}


def test_builds_what_the_agent_chose(monkeypatch):
    for key in ("DEEPGRAM_API_KEY", "CEREBRAS_API_KEY", "CARTESIA_API_KEY"):
        monkeypatch.setenv(key, "test")

    async def run():
        s = speech_config(CONFIG)
        return build("stt", s["stt"]), build("llm", s["llm"]), build("tts", s["tts"])

    stt, llm, tts = asyncio.run(run())
    assert stt.model == "nova-3" and stt._opts.language == "hi"
    assert llm.model == "gpt-oss-120b"
    assert tts.model == "sonic-3.6" and tts._opts.voice == CONFIG["tts"]["voiceId"] and tts._opts.language == "hi"


def test_provider_tags_can_differ_from_the_language():
    s = speech_config({**CONFIG, "language": "en", "stt": {"provider": "deepgram", "model": "nova-3", "language": "en-IN"}})
    assert s["stt"]["language"] == "en-IN" and s["tts"]["language"] == "hi"


def test_missing_config_uses_defaults_and_the_language():
    s = speech_config({"language": "ta"})
    assert s["stt"] == {**models.DEFAULTS["stt"], "language": "ta"}
    assert s["tts"]["provider"] == "cartesia" and s["tts"]["language"] == "ta"
    assert "language" not in s["llm"]


def test_unknown_provider_falls_back_but_keeps_the_voice():
    s = speech_config({**CONFIG, "tts": {"provider": "nope", "model": "x", "voiceId": "v1"}})
    assert s["tts"]["provider"] == "cartesia" and s["tts"]["model"] == "sonic-3.6" and s["tts"]["voiceId"] == "v1"
