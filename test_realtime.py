"""Realtime (GPT-Live) agents: built from the call config; the opening line kept, other word-for-word lines dropped."""

import asyncio

import agent
from models import build, opening_line_instructions, realtime_call_config, speech_config

CONFIG = {
    "mode": "realtime",
    "language": "hi",
    "languageName": "Hindi",
    "realtime": {"provider": "openai", "model": "gpt-live-1", "voiceId": "cinder"},
    "prompt": "You are a helpful assistant.",
    "greeting": "Namaste!",
    "endCallMessage": "Dhanyavaad!",
    "fillerWords": ["haan"],
    "script": {"steps": []},
    "silence": {"enabled": True, "timeoutSec": 10, "message": "Kya aap line par hain?"},
}


def test_builds_gpt_live_with_the_chosen_voice(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    async def run():
        return build("realtime", speech_config(CONFIG)["realtime"])

    model = asyncio.run(run())
    assert type(model).__name__ == "GPTLiveModel"
    assert model._opts.model == "gpt-live-1" and model._opts.voice == "cinder"


def test_realtime_calls_keep_the_opening_line_drop_other_lines_and_name_the_language():
    c = realtime_call_config(CONFIG)
    assert c["greeting"] == "Namaste!"
    for key in ("endCallMessage", "fillerWords", "script"):
        assert key not in c
    assert c["silence"]["message"] == "" and c["silence"]["timeoutSec"] == 10
    assert c["prompt"].startswith("You are a helpful assistant.") and c["prompt"].endswith("Speak with the caller in Hindi.")


def test_pipeline_calls_are_untouched():
    pipeline = {**CONFIG, "mode": "pipeline"}
    assert realtime_call_config(pipeline) is pipeline
    assert speech_config(pipeline)["mode"] == "pipeline"


def test_realtime_sessions_keep_default_turn_taking_and_skip_scripts():
    c = realtime_call_config(CONFIG)
    assert agent.session_options(c) == {}
    assert not agent.wants_script(c)
    a = agent.build_agent(c["prompt"], "", c, session=None)
    assert type(a).__name__ == "CallAgent" and a._filler is None


def test_builds_gemini_live_without_thinking_settings(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")  # either variable name works

    async def run():
        cfg = {**CONFIG, "realtime": {"provider": "google", "model": "gemini-3.8-live", "voiceId": "Kore"}}
        return build("realtime", speech_config(cfg)["realtime"])

    model = asyncio.run(run())
    assert type(model).__module__.startswith("livekit.plugins.google")
    assert model._opts.model == "gemini-3.8-live" and model._opts.voice == "Kore"
    assert not model._opts.thinking_config  # 3.8 rejects thinking settings


def test_opening_line_is_asked_for_verbatim():
    text = opening_line_instructions("Namaste Asha ji!")
    assert text.endswith("\n\nNamaste Asha ji!") and "word for word" in text
