"""Silence handling: when the caller goes quiet, ask whether they're still there, then hang up.

The clock starts whenever the agent finishes speaking and is waiting for the caller. If no words
are transcribed within `timeoutSec`, the agent asks (the agent's configured line, or a short
question the LLM phrases in the conversation language). After `maxPrompts` unanswered asks the
call ends, with the end-call message if one is set.

Only transcribed words count as the caller being there. Background noise can make the voice
activity detector think someone is talking, but it produces no transcript, so it neither stops
the clock nor resets the asks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from livekit.agents import AgentSession
from livekit.agents.voice.events import AgentStateChangedEvent, UserInputTranscribedEvent

from end_call import CallEnder, hang_up_after_goodbye

logger = logging.getLogger("voice-agent.silence")

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_PROMPTS = 2
ASK_INSTRUCTIONS = (
    "The caller has been silent for a while. Briefly ask whether they are still there, in the "
    "language of the conversation. One short sentence, nothing else."
)


class SilenceWatch:
    def __init__(
        self, session: AgentSession, ender: CallEnder, config: dict[str, Any] | None, goodbye: str = "", realtime: bool = False
    ):
        config = config or {}
        self.enabled = config.get("enabled", True) is not False
        self.timeout = float(config.get("timeoutSec") or DEFAULT_TIMEOUT_SECONDS)
        self.max_prompts = int(config.get("maxPrompts") or DEFAULT_MAX_PROMPTS)
        self.message = (config.get("message") or "").strip()
        self.goodbye = goodbye.strip()
        self._session = session
        self._ender = ender
        self._realtime = realtime  # realtime models take no per-reply tool choice
        self._timer: asyncio.TimerHandle | None = None
        self._live = False
        self.prompts = 0  # unanswered asks so far

    def attach(self) -> None:
        """Starts listening to the session. Call once the callee is on the line."""
        if not self.enabled:
            return
        self._live = True
        self._session.on("agent_state_changed", self._on_agent_state)
        self._session.on("user_input_transcribed", self._on_transcript)
        if self._session.agent_state == "listening":
            self._start()

    def stop(self) -> None:
        self._live = False
        self._cancel()

    def _start(self) -> None:
        self._cancel()
        if self._live and not self._ender.ending:
            self._timer = asyncio.get_running_loop().call_later(self.timeout, self._on_silence)

    def _cancel(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_agent_state(self, ev: AgentStateChangedEvent) -> None:
        if ev.new_state == "listening":
            self._start()  # the agent finished speaking: the caller's turn
        elif ev.new_state in ("thinking", "speaking"):
            self._cancel()  # the agent is busy (answering, running a tool, asking)

    def _on_transcript(self, ev: UserInputTranscribedEvent) -> None:
        if not ev.transcript.strip():
            return
        # Words mean the caller is there: restart the clock while they talk
        if self._session.agent_state == "listening":
            self._start()
        if ev.is_final:
            self.prompts = 0

    def _on_silence(self) -> None:
        self._timer = None
        if not self._live or self._ender.ending:
            return
        if self.prompts >= self.max_prompts:
            self._hang_up()
            return
        self.prompts += 1
        logger.info("caller silent for %.0f s, asking (%d/%d)", self.timeout, self.prompts, self.max_prompts)
        try:
            if self.message:
                self._session.say(self.message)
            else:
                self._session.generate_reply(instructions=ASK_INSTRUCTIONS, **({} if self._realtime else {"tool_choice": "none"}))
        except Exception:
            logger.exception("could not ask whether the caller is still there")
            self._start()
        # The ask makes the agent speak; when it's done ("listening") the clock starts again

    def _hang_up(self) -> None:
        self.stop()
        reason = f"No response from the caller after {self.max_prompts} prompt{'s' if self.max_prompts != 1 else ''}"
        logger.info("%s, ending the call", reason)
        hang_up_after_goodbye(self._session, self._ender, self.goodbye, reason)
