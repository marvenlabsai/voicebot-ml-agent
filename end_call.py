"""Lets the agent hang up by itself once the conversation is over.

Every call gets an `end_call` tool. When the LLM calls it, the agent says goodbye and the call
ends as soon as that goodbye has played. The goodbye is the agent's configured end-call message,
spoken word for word, or a short one the LLM writes when no message is set. Scripted "end" scenarios hang up through the
same CallEnder, so the backend sees one consistent reason for agent-ended calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Annotated

from livekit.agents import RunContext, function_tool
from pydantic import Field

logger = logging.getLogger("voice-agent.end-call")

# A hang-up this soon after the callee answers is almost always a model mistake
MIN_CALL_SECONDS = float(os.getenv("END_CALL_MIN_SECONDS", "8"))
# Hang up anyway if the goodbye never finishes playing (stuck TTS, lost audio track)
GOODBYE_TIMEOUT_SECONDS = 20.0
# Extra allowance for a long configured message (roughly 12 characters of speech per second)
CHARS_PER_SECOND = 12
MAX_REASON_CHARS = 120

END_CALL_DESCRIPTION = """Hang up the call. This is your final action: nothing can be said after the goodbye.

Call it when:
- the caller says goodbye or clearly has nothing more to discuss
- the purpose of the call is done (e.g. the task is complete or a callback time is agreed)
- the caller asks you to stop calling, is not interested and wants to end, or is the wrong person
- the caller is abusive or the line is clearly an answering machine / voicemail

Do not call it when:
- the caller asks you to wait, hold on, or repeat something
- you are unsure whether they want to end; ask them instead

Do not say goodbye before calling this tool: you will be asked to say it right after."""


class CallEnder:
    """Ends the call at most once, from the end_call tool or a scripted 'end' scenario."""

    def __init__(self, hang_up: Callable[[str], None], *, min_seconds: float = MIN_CALL_SECONDS):
        self._hang_up = hang_up
        self._min_seconds = min_seconds
        self._live_since: float | None = None
        self._ended = False
        # Set as soon as a hang-up is decided (the goodbye may still be playing)
        self.ending = False

    def mark_live(self) -> None:
        """The callee is on the line (phone answered / browser session started)."""
        self._live_since = time.monotonic()

    def too_early(self) -> bool:
        if self._live_since is None:
            return True  # still ringing
        return time.monotonic() - self._live_since < self._min_seconds

    def end(self, reason: str) -> None:
        """Hangs up now. Safe to call more than once."""
        if self._ended:
            return
        self._ended = True
        self.ending = True
        logger.info("agent is ending the call: %s", reason)
        self._hang_up(reason)


def end_call_tool(ender: CallEnder, goodbye: str = ""):
    """The `end_call` function tool, bound to this call's CallEnder.

    goodbye: the agent's end-call message (variables already filled). Empty = the LLM says goodbye.
    """
    goodbye = goodbye.strip()

    async def end_call(
        context: RunContext,
        reason: Annotated[
            str, Field(description='Why the call is ending, in a few words (e.g. "caller said goodbye", "callback booked")')
        ],
    ) -> str | None:
        if ender.ending:
            return "The call is already ending. Do not say anything else."
        if ender.too_early():
            logger.info("end_call ignored: the call only just started (reason=%r)", reason)
            return (
                "The call was NOT ended because it has only just started. "
                "Continue the conversation normally and do not mention this."
            )

        ender.ending = True
        why = " ".join((reason or "conversation finished").split())[:MAX_REASON_CHARS]
        try:
            context.disallow_interruptions()  # the goodbye shouldn't be cut off
        except RuntimeError:
            pass  # the turn was already interrupted; we still hang up once it's done

        def hang_up(_handle=None) -> None:
            ender.end(f"Agent ended the call: {why}")

        asyncio.get_running_loop().call_later(GOODBYE_TIMEOUT_SECONDS + len(goodbye) / CHARS_PER_SECOND, hang_up)

        if goodbye:
            # Queued behind the current turn (not awaited: that turn is waiting on this tool).
            # Returning None means the LLM doesn't add a reply of its own.
            context.session.say(goodbye, allow_interruptions=False).add_done_callback(hang_up)
            return None

        # The goodbye generated from this tool's result plays on the same speech handle, so
        # its done callback fires once the goodbye has been fully spoken.
        context.speech_handle.add_done_callback(hang_up)
        return (
            "The call will end as soon as you finish speaking. Say one short, polite goodbye "
            "in the language of the conversation and nothing else. Do not call any tool."
        )

    return function_tool(end_call, name="end_call", description=END_CALL_DESCRIPTION)
