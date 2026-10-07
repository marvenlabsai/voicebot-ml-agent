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
    # Google's default speech detection: nothing sent, Gemini decides turns itself
    assert not model._opts.realtime_input_config
    assert model.capabilities.turn_detection


def test_gemini_start_sensitivity_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = {**CONFIG, "realtime": {"provider": "google", "model": "gemini-3.8-live", "voiceId": "Kore"}}

    async def run():
        return build("realtime", speech_config(cfg)["realtime"])

    for level, expected in (("low", "START_SENSITIVITY_LOW"), ("HIGH", "START_SENSITIVITY_HIGH")):
        monkeypatch.setenv("GEMINI_START_SENSITIVITY", level)
        detection = asyncio.run(run())._opts.realtime_input_config.automatic_activity_detection
        assert detection.start_of_speech_sensitivity.value == expected and not detection.disabled
    monkeypatch.setenv("GEMINI_START_SENSITIVITY", "loud")  # unknown value: Google's default
    assert not asyncio.run(run())._opts.realtime_input_config


def test_only_pipeline_sessions_get_the_local_vad():
    vad = object()
    assert agent.session_options(CONFIG, vad) == {}  # realtime: the model detects speech itself
    pipeline = agent.session_options({**CONFIG, "mode": "pipeline"}, vad)
    assert pipeline["vad"] is vad and "turn_handling" in pipeline


def test_opening_line_is_asked_for_verbatim():
    text = opening_line_instructions("Namaste Asha ji!")
    assert text.endswith("\n\nNamaste Asha ji!") and "word for word" in text


# --- opening line: caller audio paused while a realtime model greets -------------------------

from opening import GREET_INSTRUCTIONS, speak_opening_line  # noqa: E402


class _Handle:
    def __init__(self, playout):
        self._playout = playout

    async def wait_for_playout(self):
        await self._playout()

    def __await__(self):  # like LiveKit's SpeechHandle
        return self.wait_for_playout().__await__()


class _FakeSession:
    def __init__(self, playout=None):
        self.audio = []  # set_audio_enabled calls, in order
        self.replies = []
        self.said = []
        self.input = self
        self._playout = playout or (lambda: asyncio.sleep(0))

    def set_audio_enabled(self, on):
        self.audio.append(on)

    def generate_reply(self, **kw):
        # Caller audio must already be off when the reply starts
        self.replies.append((kw, list(self.audio)))
        return _Handle(self._playout)

    async def say(self, text):
        self.said.append(text)


def test_realtime_opening_line_pauses_caller_audio_until_it_has_played():
    s = _FakeSession()
    asyncio.run(speak_opening_line(s, object(), "Namaste!", realtime=True))
    assert s.audio == [False, True]
    (kw, audio_at_start), = s.replies
    assert audio_at_start == [False]
    assert kw["instructions"].endswith("Namaste!") and "tool_choice" not in kw


def test_realtime_greeting_without_opening_line_is_protected_too():
    s = _FakeSession()
    asyncio.run(speak_opening_line(s, object(), "", realtime=True))
    assert s.audio == [False, True] and s.replies[0][0]["instructions"] == GREET_INSTRUCTIONS


def test_caller_audio_comes_back_when_playout_fails_or_hangs(monkeypatch):
    async def boom():
        raise RuntimeError("model went away")

    s = _FakeSession(boom)
    asyncio.run(speak_opening_line(s, object(), "Hi", realtime=True))
    assert s.audio == [False, True]

    import opening

    monkeypatch.setattr(opening, "OPENING_LINE_MAX_SEC", 0.05)
    s = _FakeSession(lambda: asyncio.sleep(5))
    asyncio.run(speak_opening_line(s, object(), "Hi", realtime=True))
    assert s.audio == [False, True]


def test_caller_audio_stays_off_when_the_call_is_ending():
    class Ender:
        ending = True

    s = _FakeSession()
    asyncio.run(speak_opening_line(s, object(), "Hi", realtime=True, ender=Ender()))
    assert s.audio == [False]


def test_pipeline_opening_line_is_unchanged():
    s = _FakeSession()
    asyncio.run(speak_opening_line(s, object(), "Namaste!", realtime=False))
    assert s.said == ["Namaste!"] and s.audio == []
    s = _FakeSession()
    asyncio.run(speak_opening_line(s, object(), "", realtime=False))
    assert s.audio == [] and s.replies[0][0] == {"instructions": GREET_INSTRUCTIONS, "tool_choice": "none"}
