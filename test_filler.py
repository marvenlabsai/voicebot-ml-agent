"""Filler-word filter: fillers are dropped only while the agent is audible; Hindi matches correctly."""

import asyncio

import pytest
from livekit.agents import Agent, stt

import filler
from call_agent import CallAgent
from filler import FillerMatcher, normalize

WORDS = ["haan", "ji", "hmm", "okay", "theek hai", "हाँ", "जी", "अच्छा", "hello"]


@pytest.mark.parametrize(
    "text",
    ["haan", "Haan ji.", "hmm, okay", "theek hai", "ji ji", "हाँ", "हां जी।", "अच्छा!", "Hello?", "हाँ, जी"],
)
def test_fillers_match(text):
    assert FillerMatcher(WORDS).is_filler(text)


@pytest.mark.parametrize(
    "text",
    ["", "haan but wait", "no", "hai", "theek", "haan ji haan ji haan", "हाँ मुझे बताइए", "जीत"],
)
def test_real_speech_does_not_match(text):
    assert not FillerMatcher(WORDS).is_filler(text)


def test_hindi_vowel_signs_survive_normalization():
    assert normalize("नमस्ते, जी!") == ["नमस्ते", "जी"]  # matras/virama kept, punctuation gone
    assert normalize("हाँ") == normalize("हां")  # candrabindu = anusvara


def test_empty_list_disables_the_filter():
    a = CallAgent(instructions="p")
    a.set_filler_words([])
    assert a._filler is None
    a.set_filler_words(None)
    assert a._filler is None


def _ev(text, kind=stt.SpeechEventType.FINAL_TRANSCRIPT):
    return stt.SpeechEvent(type=kind, alternatives=[stt.SpeechData(language="hi", text=text)])


def _run_stt(agent, events, monkeypatch):
    async def fake_default(_agent, _audio, _settings):
        for ev in events:
            yield ev

    monkeypatch.setattr(Agent.default, "stt_node", fake_default)

    async def run():
        return [e async for e in agent.stt_node(None, None)]

    return asyncio.run(run())


def test_fillers_dropped_only_while_agent_is_audible(monkeypatch):
    a = CallAgent(instructions="p")
    a.set_filler_words(WORDS)
    events = [
        _ev("haan", stt.SpeechEventType.INTERIM_TRANSCRIPT),
        _ev("haan ji"),
        _ev("haan, mujhe ek sawal hai"),
        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
    ]
    a._agent_audible = True
    out = _run_stt(a, events, monkeypatch)
    assert [e.type for e in out] == [stt.SpeechEventType.FINAL_TRANSCRIPT, stt.SpeechEventType.START_OF_SPEECH]
    assert out[0].alternatives[0].text == "haan, mujhe ek sawal hai"

    a._agent_audible = False  # agent silent: "haan" is an answer
    assert len(_run_stt(a, events, monkeypatch)) == 4


def test_playback_events_track_audibility():
    from livekit import rtc
    from types import SimpleNamespace

    out = rtc.EventEmitter()
    a = CallAgent(instructions="p")
    type(a).session = property(lambda self: SimpleNamespace(output=SimpleNamespace(audio=out)))
    try:
        a._watch_playback()
        out.emit("playback_started", None)
        assert a._agent_audible
        out.emit("playback_finished", None)
        assert not a._agent_audible
    finally:
        del type(a).session


def test_scripted_agent_has_the_filter():
    from scripted.agent import ScriptedAgent

    assert issubclass(ScriptedAgent, filler.FillerFilterMixin)


def test_build_agent_applies_configured_words():
    import agent

    a = agent.build_agent("p", "", {"fillerWords": ["haan"]}, session=None)
    assert a._filler.is_filler("haan") and not a._filler.is_filler("hmm")
