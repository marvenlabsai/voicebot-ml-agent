"""No speech-to-text while the phone rings: STT opens and caller audio flows only after answer."""

import asyncio
from types import SimpleNamespace

from livekit.agents import Agent, stt

import agent as agent_module
from call_agent import CallAgent


def test_stt_waits_for_answer(monkeypatch):
    opened = []

    async def fake_default(_agent, _audio, _settings):
        opened.append(True)  # the STT connection would open here
        yield stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)

    monkeypatch.setattr(Agent.default, "stt_node", fake_default)

    async def run():
        a = CallAgent(instructions="p")
        a.hold_stt_until_answered()
        events = []

        async def consume():
            async for ev in a.stt_node(None, None):
                events.append(ev)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert opened == [] and events == []  # still ringing
        a.callee_answered()
        await task
        return opened, events

    opened, events = asyncio.run(run())
    assert opened == [True] and len(events) == 1


def test_browser_calls_are_not_held(monkeypatch):
    async def fake_default(_agent, _audio, _settings):
        yield stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)

    monkeypatch.setattr(Agent.default, "stt_node", fake_default)

    async def run():
        return [e async for e in CallAgent(instructions="p").stt_node(None, None)]

    assert len(asyncio.run(run())) == 1


class _Reporter:
    def __init__(self, log):
        self.log = log

    async def send(self, kind, **_):
        self.log.append(f"report:{kind}")


def _dial(log, *, fail=False):
    """Runs dial() with fakes and returns the order things happened in."""
    from livekit import api

    agent = CallAgent(instructions="p")
    agent.hold_stt_until_answered = lambda: log.append("stt held")
    agent.callee_answered = lambda: log.append("stt released")

    async def start(**_):
        log.append("session started")

    session = SimpleNamespace(
        start=start,
        input=SimpleNamespace(set_audio_enabled=lambda on: log.append(f"audio {'on' if on else 'off'}")),
    )

    async def create_sip_participant(_req):
        log.append("ringing…")
        await asyncio.sleep(0.02)
        if fail:
            raise api.ServerError("busy")  # type: ignore[call-arg]
        log.append("answered")

    ctx = SimpleNamespace(
        room=SimpleNamespace(name="r", on=lambda *_: None),
        api=SimpleNamespace(sip=SimpleNamespace(create_sip_participant=create_sip_participant)),
    )
    config = {"phoneNumber": "+911234567890", "sipTrunkId": "ST_1"}
    state = {"failed": False, "answered": False}
    ok = asyncio.run(agent_module.dial(ctx, session, agent, config, _Reporter(log), state))
    return ok, state


def test_dial_releases_stt_only_after_answer():
    log = []
    ok, state = _dial(log)
    assert ok and state["answered"]
    assert log.index("audio off") < log.index("ringing…")
    assert log.index("stt held") < log.index("ringing…")
    assert log.index("answered") < log.index("audio on") < log.index("report:answered")
    assert log.index("answered") < log.index("stt released")


def test_unanswered_call_never_opens_stt(monkeypatch):
    from livekit import api

    monkeypatch.setattr(api, "ServerError", type("ServerError", (Exception,), {"message": "busy"}))
    log = []
    ok, state = _dial(log, fail=True)
    assert not ok and state["failed"]
    assert "stt released" not in log and "audio on" not in log
