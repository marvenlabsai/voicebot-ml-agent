"""ScriptedAgent: answers matched turns with pre-synthesized lines, everything else with the LLM."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from livekit.agents import Agent, StopResponse, llm

from .reply_audio import ReplyAudio
from .router import ScriptRouter

logger = logging.getLogger("voice-agent.script")

# A match that takes longer than this is abandoned and the LLM answers instead
MATCH_TIMEOUT = int(os.getenv("SCRIPT_MATCH_TIMEOUT_MS", "150")) / 1000


class ScriptedAgent(Agent):
    def __init__(self, *, instructions: str, router: ScriptRouter, audio: ReplyAudio):
        super().__init__(instructions=instructions)
        self._router = router
        self._audio = audio
        self._ready = False
        self.step = router.start_step
        # Transcript tags, read by the entrypoint's conversation_item_added handler
        self._user_tags: dict[str, dict] = {}
        self._reply_tags: list[tuple[str, dict]] = []

    def start(self, greeting: str) -> None:
        """Embeds the examples and pre-synthesizes every line in the background."""
        self._audio.start([greeting, *self._router.replies()] if greeting else self._router.replies())

        async def prepare() -> None:
            try:
                await asyncio.to_thread(self._router.prepare)
                self._ready = True
            except Exception:
                logger.exception("could not prepare the script; this call will use the LLM only")

        self._prepare_task = asyncio.create_task(prepare())

    async def on_exit(self) -> None:
        await self._audio.aclose()

    def say_line(self, text: str, scenario_id: str = "greeting"):
        """Speaks a scripted line from the cache, or synthesizes it live if it isn't ready."""
        audio = self._audio.play(text)
        self._reply_tags.append((text, {"source": "script", "scenarioId": scenario_id}))
        if audio is None:
            return self.session.say(text)
        return self.session.say(text, audio=audio)

    def transcript_tag(self, item: llm.ChatMessage) -> dict:
        """Extra transcript fields for a conversation item: which source produced it."""
        if item.role == "user":
            return self._user_tags.pop(item.id, {})
        text = (item.text_content or "").strip()
        for i, (line, tag) in enumerate(self._reply_tags):
            # An interrupted line is stored truncated
            if text and line.startswith(text[:40]):
                del self._reply_tags[i]
                return tag
        return {"source": "llm"}

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        text = (new_message.text_content or "").strip()
        if not self._ready or not text:
            return  # LLM answers

        start = time.perf_counter()
        try:
            match, best = await asyncio.wait_for(self._router.match(text, self.step), MATCH_TIMEOUT)
        except Exception as e:  # timeout or classifier error: never block the call on the script
            logger.warning("script match failed (%s); falling back to the LLM", e or type(e).__name__)
            return
        took_ms = (time.perf_counter() - start) * 1000

        if match is None:
            logger.info("script fallback step=%s best=%.2f ms=%.0f text=%r", self.step, best, took_ms, text[:80])
            return

        sc = match.scenario
        logger.info(
            "script match step=%s scenario=%s score=%.2f runner_up=%.2f ms=%.0f",
            self.step, sc.id, match.score, match.runner_up, took_ms,
        )

        # StopResponse skips the framework's own commit of the user's message, so add it here:
        # the LLM keeps the full conversation for later fallbacks, and the transcript has it.
        self._user_tags[new_message.id] = {"source": "script", "scenarioId": sc.id, "score": round(match.score, 3)}
        self._commit_user_message(new_message)

        handle = self.say_line(sc.reply, sc.id)
        if sc.next == "end":
            asyncio.create_task(self._end_after(handle))
        elif sc.next != "stay":
            self.step = sc.next
        raise StopResponse()

    def _commit_user_message(self, message: llm.ChatMessage) -> None:
        try:
            # Same as what AgentActivity does for a normal turn
            self._chat_ctx.items.append(message)
            self.session._conversation_item_added(message)
        except Exception:
            logger.warning("could not add the user's message to the chat history", exc_info=True)

    async def _end_after(self, handle) -> None:
        try:
            await handle
        finally:
            # Closing the session runs the entrypoint's close handler, which hangs up
            self.session.shutdown(drain=True)
