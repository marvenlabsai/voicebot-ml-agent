"""Scripted replies (opt-in). Nothing here is imported unless a call actually uses a script.

Enabled only when SCRIPTED_REPLIES=1 in the agent's env *and* the dispatch metadata carries an
enabled script. Any problem setting it up falls back to the plain LLM agent.
"""

from __future__ import annotations

import logging

from livekit.agents import Agent, AgentSession

logger = logging.getLogger("voice-agent.script")


def build_scripted_agent(*, prompt: str, script: dict, greeting: str, session: AgentSession, config: dict) -> Agent:
    from .agent import ScriptedAgent
    from .reply_audio import ReplyAudio
    from .router import ScriptRouter

    tts = session.tts
    if tts is None:
        raise RuntimeError("the session has no TTS")
    cache_key = "|".join(
        str(x) for x in (tts.provider, tts.model, config.get("voiceId") or "default", config.get("language") or "en")
    )
    agent = ScriptedAgent(instructions=prompt, router=ScriptRouter(script), audio=ReplyAudio(tts, cache_key))
    agent.start(greeting)
    logger.info(
        "scripted replies on: %d steps, %d global scenarios",
        len(script.get("steps") or []),
        len(script.get("global") or []),
    )
    return agent
