"""Tests for scripted replies. Run from agent/: python -m pytest test_scripted.py -q"""

import subprocess
import sys
import time

import pytest

SCRIPT = {
    "threshold": 0.75,
    "steps": [
        {
            "id": "start",
            "scenarios": [
                {"id": "yes", "examples": ["yes", "yeah sure", "haan", "haan ji", "ji haan", "bilkul", "हाँ", "हाँ जी", "okay go ahead", "theek hai"], "reply": "Great!", "next": "details"},
                {"id": "no", "examples": ["no", "nahi", "nahi chahiye", "नहीं", "not interested", "no thanks", "mujhe nahi chahiye", "rehne do"], "reply": "No problem, have a nice day.", "next": "end"},
                {"id": "busy", "examples": ["I'm busy right now", "call me later", "abhi busy hu", "baad mein call karo", "बाद में कॉल करना", "I'm driving", "meeting mein hu"], "reply": "Sure, I'll call later.", "next": "end"},
            ],
        },
        {"id": "details", "scenarios": [{"id": "price", "examples": ["how much does it cost", "what's the price", "kitne ka hai", "price kya hai", "कितने का है", "charges kitne hain"], "reply": "It's 499 a month."}]},
    ],
    "global": [
        {"id": "who", "examples": ["who is this", "who are you", "aap kaun", "kaun bol raha hai", "कौन बोल रहा है", "where are you calling from"], "reply": "I'm calling from Acme."},
    ],
}


@pytest.fixture(scope="module")
def router():
    from scripted.router import ScriptRouter

    r = ScriptRouter(SCRIPT)
    r.prepare()
    return r


@pytest.mark.parametrize(
    "text,step,expected",
    [
        ("haan bilkul bataiye", "start", "yes"),
        ("हाँ बोलिए", "start", "yes"),
        ("ok ok theek hai bataiye", "start", "yes"),
        ("who are you", "start", "who"),  # global scenarios match from any step
        ("what's the price", "details", "price"),
    ],
)
def test_routes(router, text, step, expected):
    match, _ = router.match_sync(text, step)
    assert match is not None and match.scenario.id == expected


@pytest.mark.parametrize("text", ["what's the weather in delhi", "can you write me a poem about cricket"])
def test_unrelated_text_falls_back(router, text):
    assert router.match_sync(text, "start")[0] is None


def test_only_current_step_scenarios(router):
    # "price" lives in step "details", so it can't match while on "start"
    match, _ = router.match_sync("what's the price", "start")
    assert match is None or match.scenario.id != "price"


def test_never_misroutes_hard_hinglish(router):
    # Too ambiguous for the small model: these must fall back, never pick a wrong scenario
    expected = {"nahi abhi nahi": "no", "aapko mera number kahan se mila": "who", "abhi gaadi chala raha hu": "busy"}
    for text, right in expected.items():
        match, _ = router.match_sync(text, "start")
        assert match is None or match.scenario.id == right, (text, match and match.scenario.id)


def test_match_is_fast(router):
    router.match_sync("warm up", "start")
    start = time.perf_counter()
    for _ in range(20):
        router.match_sync("haan bilkul bataiye", "start")
    assert (time.perf_counter() - start) / 20 < 0.05


def test_replies_listed_for_presynthesis(router):
    assert set(router.replies()) == {"Great!", "No problem, have a nice day.", "Sure, I'll call later.", "It's 499 a month.", "I'm calling from Acme."}


def test_plain_agent_never_loads_script_code():
    """Isolation: without SCRIPTED_REPLIES=1, importing and building the agent must not touch fastembed."""
    code = (
        "import os, sys; os.environ.pop('SCRIPTED_REPLIES', None);"
        "import agent;"
        "a = agent.build_agent('p', 'hi', {'script': {'steps': []}}, session=None);"
        "print(type(a).__name__, 'fastembed' in sys.modules, 'scripted' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"SCRIPTED_REPLIES": "0", **_base_env()})
    assert out.stdout.split()[-3:] == ["Agent", "False", "False"], out.stderr[-2000:]


def test_broken_script_falls_back_to_plain_agent():
    """Isolation: a script that blows up during setup must leave a working LLM agent."""
    code = (
        "import agent;"
        "a = agent.build_agent('p', 'hi', {'script': {'steps': 'garbage'}}, session=None);"
        "print(type(a).__name__)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**_base_env(), "SCRIPTED_REPLIES": "1"})
    assert out.stdout.strip().splitlines()[-1] == "Agent", out.stderr[-2000:]


def _base_env():
    import os

    return {k: v for k, v in os.environ.items() if k != "SCRIPTED_REPLIES"}


class _FakeSession:
    def __init__(self):
        self.said, self.items, self.closed = [], [], False

    def say(self, text, audio=None):
        self.said.append((text, audio is not None))
        fut = __import__("asyncio").get_running_loop().create_future()
        fut.set_result(None)
        return fut

    def _conversation_item_added(self, msg):
        self.items.append(msg)

    def shutdown(self, drain=True):
        self.closed = True


class _FakeAudio:
    def __init__(self, ready=()):
        self.ready = set(ready)

    def start(self, lines):
        pass

    def play(self, text):
        return iter(()) if text in self.ready else None


def _scripted_agent(router, audio=None):
    from scripted.agent import ScriptedAgent

    session = _FakeSession()

    class TestAgent(ScriptedAgent):
        @property
        def session(self):
            return session

    a = TestAgent(instructions="p", router=router, audio=audio or _FakeAudio({"Great!"}))
    a._ready = True
    return a, session


def test_turn_matched_plays_line_and_moves_step(router):
    import asyncio

    from livekit.agents import StopResponse, llm

    a, session = _scripted_agent(router)
    msg = llm.ChatMessage(role="user", content=["haan bilkul bataiye"])
    with pytest.raises(StopResponse):
        asyncio.run(a.on_user_turn_completed(llm.ChatContext.empty(), msg))
    assert session.said == [("Great!", True)]  # cached audio used
    assert a.step == "details"
    assert session.items == [msg] and a.chat_ctx.items[-1].id == msg.id  # caller's words kept
    assert a.transcript_tag(msg)["scenarioId"] == "yes"
    assert a.transcript_tag(llm.ChatMessage(role="assistant", content=["Great!"])) == {"source": "script", "scenarioId": "yes"}
    assert a.transcript_tag(llm.ChatMessage(role="assistant", content=["LLM words"])) == {"source": "llm"}


def test_turn_unmatched_goes_to_llm(router):
    import asyncio

    from livekit.agents import llm

    a, session = _scripted_agent(router)
    asyncio.run(a.on_user_turn_completed(llm.ChatContext.empty(), llm.ChatMessage(role="user", content=["tell me a joke about cricket"])))
    assert session.said == [] and a.step == "start"


def test_end_scenario_hangs_up_and_uncached_line_is_live(router):
    import asyncio

    from livekit.agents import StopResponse, llm

    a, session = _scripted_agent(router, _FakeAudio())

    async def run():
        with pytest.raises(StopResponse):
            await a.on_user_turn_completed(llm.ChatContext.empty(), llm.ChatMessage(role="user", content=["nahi mujhe nahi chahiye"]))
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert session.said == [("No problem, have a nice day.", False)]  # not cached → live TTS
    assert session.closed


def test_classifier_error_goes_to_llm(router):
    import asyncio

    from livekit.agents import llm

    a, session = _scripted_agent(router)

    async def boom(*_):
        raise RuntimeError("model crashed")

    a._router = type("R", (), {"match": staticmethod(boom), "start_step": "start"})()
    asyncio.run(a.on_user_turn_completed(llm.ChatContext.empty(), llm.ChatMessage(role="user", content=["haan"])))
    assert session.said == []


def test_reply_audio_cache_roundtrip(tmp_path, monkeypatch):
    import asyncio

    from livekit import rtc

    from scripted import reply_audio

    monkeypatch.setattr(reply_audio, "CACHE_DIR", tmp_path)
    calls = []

    class FakeStream:
        def __init__(self, text):
            self.text = text

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def collect(self):
            calls.append(self.text)
            n = 24000 // 2  # 0.5 s of 24 kHz mono
            return rtc.AudioFrame(bytes(range(256)) * (n * 2 // 256) + bytes(n * 2 % 256), 24000, 1, n)

    class FakeTTS:
        def synthesize(self, text):
            return FakeStream(text)

    async def run():
        first = reply_audio.ReplyAudio(FakeTTS(), "v1")
        first.start(["Hello", "Hello", "Bye"])
        await first._task
        frames = [f async for f in first.play("Hello")]
        assert len(frames) == 25 and all(f.samples_per_channel == 480 for f in frames)  # 20 ms frames
        assert first.play("missing") is None

        second = reply_audio.ReplyAudio(FakeTTS(), "v1")  # a later call: served from disk
        second.start(["Hello", "Bye"])
        await second._task
        frames2 = [f async for f in second.play("Hello")]
        assert b"".join(bytes(f.data) for f in frames2) == b"".join(bytes(f.data) for f in frames)

    asyncio.run(run())
    assert sorted(calls) == ["Bye", "Hello"]  # each line synthesized once


def test_preemptive_generation_off_only_for_scripted_agents():
    """Scripted agents disable preemptive generation; plain agents keep the framework default."""
    code = (
        "import agent;"
        "from livekit.agents import AgentSession;"
        "on = lambda cfg: AgentSession(**agent.session_options(cfg)).options.preemptive_generation['enabled'];"
        "print(on({'script': {'steps': []}}), on({}), agent.session_options({}))"
    )
    scripted = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**_base_env(), "SCRIPTED_REPLIES": "1"})
    assert scripted.stdout.split() == ["False", "True", "{}"], scripted.stderr[-2000:]
    plain = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**_base_env(), "SCRIPTED_REPLIES": "0"})
    assert plain.stdout.split() == ["True", "True", "{}"], plain.stderr[-2000:]  # flag off: script ignored
