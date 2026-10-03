"""Output guard: raw output is never spoken; one regeneration; tidy text for TTS; notes logged."""

import asyncio
from types import SimpleNamespace

import pytest
from livekit.agents import Agent, llm

from call_agent import CallAgent
from output_guard import RETRY_INSTRUCTIONS, TextGuard, tidy


def _stream(guard, chunks):
    out = []
    for c in chunks:
        out += guard.feed(c)
        if guard.hit:
            return out
    return out + guard.finish()


def test_plain_speech_passes_through_tidied():
    g = TextGuard(strict=True)
    out = _stream(g, ["Your order ", "is on the way.", "It will reach", " on Monday!\nKya aur", " kuch?"])
    assert g.hit is None
    assert "".join(out) == "Your order is on the way. It will reach on Monday! Kya aur kuch? "


@pytest.mark.parametrize(
    "chunks,kind",
    [
        (["Here is the result: ", '{"status": "shipped"}'], "JSON"),
        (["Okay. <function=end_call>{}</function>"], "tag"),
        (["Calling now to=functions.get_order with id 5."], "tool call syntax"),
        (["SUCCESS: The get_order action completed."], "tool result"),
        (["Success (HTTP 200). Response: shipped."], "tool result"),
        (['The "status": "shipped" field says so.'], "JSON"),
        (["Traceback (most recent call last): oops."], "stack trace"),
        (["UnauthorizedException: token expired."], "error text"),
        (["```json\nhi\n```"], "code"),
    ],
)
def test_raw_output_is_caught(chunks, kind):
    g = TextGuard(strict=True)
    out = _stream(g, chunks)
    assert g.hit and g.hit[0] == kind
    assert not any("{" in o or "<" in o or "SUCCESS" in o or "HTTP" in o for o in out)


def test_preamble_before_json_is_not_spoken():
    g = TextGuard(strict=True)
    out = _stream(g, ["Sure. ", "The result is ", '{"a": 1}'])
    assert out == ["Sure. "] and g.hit[1].startswith("The result is {")


def test_lenient_mode_cuts_raw_parts_and_keeps_going():
    g = TextGuard(strict=False)
    out = _stream(g, ['Done! {"id": 7} ', "SUCCESS: saved. ", "Your booking is confirmed."])
    assert g.hit is None and "".join(out).strip() == "Done! Your booking is confirmed."
    assert len(g.removed) == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("know.Kya", "know. Kya"),
        ("है।क्या", "है। क्या"),
        ("costs 3.5 lakh", "costs 3.5 lakh"),
        ("at 5:30 pm", "at 5:30 pm"),
        ("**Great** news", "Great news"),
        ("line\nbreak.", "line break. "),
    ],
)
def test_tidy(text, expected):
    assert tidy(text) == expected


class _Session:
    def __init__(self):
        self.replies = []

    def generate_reply(self, instructions=None):
        self.replies.append(instructions)


def _agent(monkeypatch, replies):
    """A CallAgent whose LLM produces `replies` (lists of chunks), one per llm_node call."""
    session, notes = _Session(), []
    a = CallAgent(instructions="p")
    a.log_transcript = notes.append
    monkeypatch.setattr(CallAgent, "session", property(lambda self: session), raising=False)
    queue = list(replies)

    async def fake_llm(_agent, _ctx, _tools, _settings):
        for c in queue.pop(0):
            yield c

    monkeypatch.setattr(Agent.default, "llm_node", fake_llm)

    async def run_turn():
        return [c async for c in a.llm_node(None, [], None)]

    return a, session, notes, run_turn


def test_hit_regenerates_once_and_notes_it(monkeypatch):
    tool_call = llm.ChatChunk(
        id="1", delta=llm.ChoiceDelta(tool_calls=[llm.FunctionToolCall(name="get_order", arguments="{}", call_id="c1")])
    )
    a, session, notes, run_turn = _agent(
        monkeypatch,
        [
            ["Let me check. ", '{"order": ', '"shipped"}', " more text"],
            ["It ships Monday. ", '{"x": 1}', " Anything else?"],
            [tool_call, "Checking."],
        ],
    )
    first = asyncio.run(run_turn())
    assert first == ["Let me check. "]
    assert session.replies == [RETRY_INSTRUCTIONS]
    assert notes[0]["role"] == "note" and notes[0]["note"]["kind"] == "output_guard"
    assert notes[0]["note"]["detail"].startswith('{"order": "shipped"}')

    second = asyncio.run(run_turn())  # the regenerated reply: cut, not regenerated again
    assert "".join(second).strip() == "It ships Monday. Anything else?"
    assert session.replies == [RETRY_INSTRUCTIONS] and notes[1]["text"] == "Raw output removed from the reply"

    third = asyncio.run(run_turn())  # back to strict; tool calls pass through untouched
    assert third[0] is tool_call and third[1] == "Checking. "


def test_filler_drop_is_noted_but_not_passed_on(monkeypatch):
    from livekit.agents import stt

    a = CallAgent(instructions="p")
    notes = []
    a.log_transcript = notes.append
    a.set_filler_words(["haan"])
    a._agent_audible = True
    ev = stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[stt.SpeechData(language="hi", text="Haan.")])
    interim = stt.SpeechEvent(type=stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[stt.SpeechData(language="hi", text="Haan")])
    assert a._should_drop(interim) and a._should_drop(ev)
    assert len(notes) == 1 and notes[0]["note"]["kind"] == "filler" and "Haan." in notes[0]["text"]
