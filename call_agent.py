"""The behaviours every agent gets, plain or scripted."""

import asyncio

from livekit.agents import Agent

from filler import FillerFilterMixin
from output_guard import OutputGuardMixin


class AnswerGateMixin:
    """Keeps the speech-to-text connection closed until the callee answers.

    Outbound sessions start while the phone is still ringing so the agent is ready the moment
    someone picks up. Without this, the STT stream (Deepgram bills for it) would open at session
    start and could hear the ringback tone or carrier messages.
    """

    _answered: asyncio.Event | None = None

    def hold_stt_until_answered(self) -> None:
        self._answered = asyncio.Event()

    def callee_answered(self) -> None:
        if self._answered is not None:
            self._answered.set()

    async def stt_node(self, audio, model_settings):
        if self._answered is not None:
            await self._answered.wait()
        async for ev in super().stt_node(audio, model_settings):  # type: ignore[misc]
            yield ev


class CallAgentMixin(OutputGuardMixin, AnswerGateMixin, FillerFilterMixin):
    """Raw-output guard + STT held until answer + filler-word filter. Put before `Agent`."""


class CallAgent(CallAgentMixin, Agent):
    """The plain LLM agent."""
