"""The end_call tool: the agent hangs up by itself, once, never right after the call starts."""

import asyncio

from end_call import CallEnder, end_call_tool


class _FakeHandle:
    def __init__(self):
        self.callbacks = []
        self.allow_interruptions = True

    def add_done_callback(self, cb):
        self.callbacks.append(cb)

    def finish(self):
        for cb in self.callbacks:
            cb(self)


class _FakeSession:
    def __init__(self):
        self.said = []

    def say(self, text, allow_interruptions=True):
        handle = _FakeHandle()
        self.said.append((text, allow_interruptions, handle))
        return handle


class _FakeContext:
    def __init__(self):
        self.speech_handle = _FakeHandle()
        self.session = _FakeSession()

    def disallow_interruptions(self):
        self.speech_handle.allow_interruptions = False


def _ender(min_seconds=0.0):
    hung_up = []
    ender = CallEnder(hung_up.append, min_seconds=min_seconds)
    return ender, hung_up


def _call(tool, ctx, reason="caller said goodbye"):
    return asyncio.run(_run(tool, ctx, reason))


async def _run(tool, ctx, reason):
    return await tool(ctx, reason=reason)


def test_tool_schema():
    from livekit.agents import llm

    tool = end_call_tool(_ender()[0])
    assert tool.info.name == "end_call"
    schema = llm.utils.build_strict_openai_schema(tool)
    assert list(schema["function"]["parameters"]["properties"]) == ["reason"]


def test_hangs_up_after_goodbye_plays():
    ender, hung_up = _ender()
    ender.mark_live()
    ctx = _FakeContext()
    out = _call(end_call_tool(ender), ctx)
    assert "goodbye" in out.lower()
    assert ender.ending and not ctx.speech_handle.allow_interruptions
    assert hung_up == []  # still saying goodbye
    ctx.speech_handle.finish()
    assert hung_up == ["Agent ended the call: caller said goodbye"]


def test_configured_message_is_spoken_verbatim_then_hangs_up():
    ender, hung_up = _ender()
    ender.mark_live()
    ctx = _FakeContext()
    out = _call(end_call_tool(ender, "  Thanks Asha, have a great day!  "), ctx)
    assert out is None  # no extra LLM goodbye
    [(text, interruptible, handle)] = ctx.session.said
    assert text == "Thanks Asha, have a great day!" and not interruptible
    assert ctx.speech_handle.callbacks == [] and hung_up == []
    handle.finish()
    assert hung_up == ["Agent ended the call: caller said goodbye"]


def test_configured_message_not_spoken_when_too_early():
    ender, hung_up = _ender(min_seconds=60)
    ender.mark_live()
    ctx = _FakeContext()
    assert "not ended" in _call(end_call_tool(ender, "Bye!"), ctx).lower()
    assert ctx.session.said == [] and hung_up == []


def test_agent_passes_the_message_to_the_tool():
    import agent

    ender, _ = _ender()
    a = agent.build_agent("p", "", {"endCallMessage": "Bye!"}, session=None, ender=ender)
    ender.mark_live()
    ctx = _FakeContext()
    assert _call(a.tools[0], ctx) is None
    assert ctx.session.said[0][0] == "Bye!"


def test_too_early_is_ignored():
    ender, hung_up = _ender(min_seconds=60)
    tool = end_call_tool(ender)
    assert "not ended" in _call(tool, _FakeContext()).lower()  # still ringing
    ender.mark_live()
    ctx = _FakeContext()
    assert "not ended" in _call(tool, ctx).lower()  # just answered
    assert not ender.ending and ctx.speech_handle.callbacks == [] and hung_up == []


def test_second_call_and_repeat_end_are_noops():
    ender, hung_up = _ender()
    ender.mark_live()
    tool = end_call_tool(ender)
    first = _FakeContext()
    _call(tool, first)
    second = _FakeContext()
    assert "already ending" in _call(tool, second).lower()
    first.speech_handle.finish()
    first.speech_handle.finish()
    ender.end("again")
    assert len(hung_up) == 1


def test_reason_is_cleaned_and_capped():
    ender, hung_up = _ender()
    ender.mark_live()
    ctx = _FakeContext()
    _call(end_call_tool(ender), ctx, reason="  callback\n booked  " + "x" * 300)
    ctx.speech_handle.finish()
    assert hung_up[0].startswith("Agent ended the call: callback booked x")
    assert len(hung_up[0]) <= len("Agent ended the call: ") + 120


def test_every_plain_agent_gets_the_tool():
    import agent

    ender, _ = _ender()
    a = agent.build_agent("p", "", {}, session=None, ender=ender)
    assert [t.info.name for t in a.tools] == ["end_call"]
