"""Silence handling: ask after the configured silence, up to N times, then hang up.
Only transcribed words count as the caller being there."""

import asyncio

from livekit import rtc
from livekit.agents.voice.events import AgentStateChangedEvent, UserInputTranscribedEvent

from end_call import CallEnder
from silence import SilenceWatch

T = 0.2  # timeout used in these tests (seconds)


class _Handle:
    def __init__(self):
        self.callbacks = []

    def add_done_callback(self, cb):
        self.callbacks.append(cb)


class _Session(rtc.EventEmitter):
    """Speaks instantly: say() goes speaking → listening, like a real session."""

    def __init__(self):
        super().__init__()
        self.agent_state = "listening"
        self.said, self.generated = [], []
        self.last_handle = None

    def _speak(self):
        self.set_state("speaking")
        asyncio.get_running_loop().call_soon(self.set_state, "listening")

    def say(self, text, allow_interruptions=True):
        self.said.append(text)
        self.last_handle = _Handle()
        self._speak()
        return self.last_handle

    def generate_reply(self, instructions=None, tool_choice=None):
        self.generated.append(tool_choice)
        self._speak()

    def set_state(self, state):
        old, self.agent_state = self.agent_state, state
        self.emit("agent_state_changed", AgentStateChangedEvent(old_state=old, new_state=state))

    def hear(self, text, final=True):
        self.emit("user_input_transcribed", UserInputTranscribedEvent(transcript=text, is_final=final))


def _setup(config=None, goodbye=""):
    ended = []
    ender = CallEnder(ended.append, min_seconds=0)
    ender.mark_live()
    session = _Session()
    watch = SilenceWatch(session, ender, {"timeoutSec": T, "maxPrompts": 2, "message": "Are you there?", **(config or {})}, goodbye)
    watch.timeout = T  # the schema minimum is whole seconds; tests run faster
    return session, watch, ended


def test_asks_twice_then_hangs_up():
    async def run():
        session, watch, ended = _setup()
        watch.attach()
        await asyncio.sleep(T * 5)
        return session, ended

    session, ended = asyncio.run(run())
    assert session.said == ["Are you there?", "Are you there?"]
    assert ended == ["No response from the caller after 2 prompts"]


def test_goodbye_is_spoken_before_hanging_up():
    async def run():
        session, watch, ended = _setup(goodbye="Bye for now!")
        watch.attach()
        await asyncio.sleep(T * 5)
        assert ended == []  # still saying goodbye
        session.last_handle.callbacks[0](session.last_handle)
        return session, ended

    session, ended = asyncio.run(run())
    assert session.said[-1] == "Bye for now!" and len(ended) == 1


def test_words_reset_the_asks_but_noise_does_not():
    async def run():
        session, watch, ended = _setup()
        watch.attach()
        await asyncio.sleep(T * 1.5)  # first ask
        assert watch.prompts == 1
        session.emit("user_state_changed", object())  # VAD noise: ignored
        session.hear("haan main hoon")  # real words
        assert watch.prompts == 0
        await asyncio.sleep(T * 0.5)
        session.set_state("thinking")  # the agent answers…
        await asyncio.sleep(T * 3)  # …for a long time: no ask while it's busy
        assert session.said == ["Are you there?"]
        session.set_state("listening")
        await asyncio.sleep(T * 1.5)
        return session, watch, ended

    session, watch, ended = asyncio.run(run())
    assert session.said == ["Are you there?", "Are you there?"] and watch.prompts == 1 and ended == []


def test_interim_words_hold_off_the_ask():
    async def run():
        session, watch, _ = _setup()
        watch.attach()
        for _ in range(6):  # the caller keeps talking (interim transcripts) past the timeout
            await asyncio.sleep(T * 0.4)
            session.hear("so what I wanted", final=False)
        return session

    assert asyncio.run(run()).said == []


def test_without_message_the_llm_asks():
    async def run():
        session, watch, _ = _setup({"message": ""})
        watch.attach()
        await asyncio.sleep(T * 1.5)
        return session

    session = asyncio.run(run())
    assert session.said == [] and session.generated == ["none"]


def test_disabled_or_call_ending_never_asks():
    async def run():
        session, watch, _ = _setup({"enabled": False})
        watch.attach()
        session2, watch2, _ = _setup()
        watch2.attach()
        watch2._ender.ending = True  # end_call already in progress
        await asyncio.sleep(T * 3)
        return session, session2

    s1, s2 = asyncio.run(run())
    assert s1.said == [] and s2.said == []
