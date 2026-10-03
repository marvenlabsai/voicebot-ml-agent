"""Turn-taking settings reach a real AgentSession and Silero VAD as intended."""

import asyncio

from livekit.agents import AgentSession
from livekit.plugins import silero

import agent
from turn_taking import endpointing, turn_handling, vad_options


def test_plain_session_options():
    async def make():
        return AgentSession(**agent.session_options({})).options

    opts = asyncio.run(make())
    assert opts.endpointing["min_delay"] == 0.2 and opts.endpointing["max_delay"] == 2.0
    i = opts.interruption
    assert (i["mode"], i["min_words"], i["min_duration"]) == ("vad", 2, 0.5)
    assert i["false_interruption_timeout"] == 1.0 and i["resume_false_interruption"]
    p = opts.preemptive_generation
    assert p["enabled"] and p["preemptive_tts"]


def test_scripted_agents_skip_preemptive_generation():
    p = turn_handling("deepgram", scripted=True)["preemptive_generation"]
    assert not p["enabled"] and not p["preemptive_tts"]


def test_endpointing_per_provider_and_env_override(monkeypatch):
    assert endpointing("deepgram")["min_delay"] == 0.2
    assert endpointing("someone-else") == {"mode": "fixed", "min_delay": 0.3, "max_delay": 2.5}
    monkeypatch.setenv("ENDPOINTING_MAX_SECONDS", "1.5")
    assert endpointing("deepgram")["max_delay"] == 1.5


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("INTERRUPTION_MIN_WORDS", "1")
    monkeypatch.setenv("PREEMPTIVE_GENERATION", "0")
    th = turn_handling("deepgram")
    assert th["interruption"]["min_words"] == 1 and not th["preemptive_generation"]["enabled"]


def test_vad_loads_with_custom_settings():
    vad = silero.VAD.load(**vad_options())
    assert vad._opts.min_silence_duration == 0.25 and vad._opts.min_speech_duration == 0.05
    assert vad._opts.activation_threshold == 0.5
