"""The agent's first words when a call connects.

People usually say "Hello?" the moment a call connects, which is exactly when a realtime model is
starting its opening line. A realtime model's own speech detection (and the session's VAD) take
that as the caller barging in and cut the greeting off, and a realtime model can't mark one reply
as uninterruptible. So for realtime agents the caller's audio is paused while the opening line
plays: anything said over it is ignored, and barge-in works as usual once it has finished.

Pipeline agents speak the line with their own TTS and are left as they were.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from models import opening_line_instructions

logger = logging.getLogger("voice-agent")

# The opening line never takes this long; caller audio comes back regardless after it
OPENING_LINE_MAX_SEC = 20.0

GREET_INSTRUCTIONS = "Greet the user briefly and ask how you can help, in the conversation language."


async def speak_opening_line(session: Any, agent: Any, greeting: str, *, realtime: bool, ender: Any = None) -> None:
    """Says the opening line (or lets the model greet when there is none)."""
    if not realtime:
        if greeting:
            say_line = getattr(agent, "say_line", None)  # scripted agents play the cached greeting
            await (say_line(greeting) if say_line else session.say(greeting))
        else:
            # The opening line must never hang up
            await session.generate_reply(instructions=GREET_INSTRUCTIONS, tool_choice="none")
        return

    instructions = opening_line_instructions(greeting) if greeting else GREET_INSTRUCTIONS
    session.input.set_audio_enabled(False)
    logger.info("caller audio paused for the opening line")
    try:
        handle = session.generate_reply(instructions=instructions)
        await asyncio.wait_for(handle.wait_for_playout(), OPENING_LINE_MAX_SEC)
    except asyncio.TimeoutError:
        logger.warning("opening line still playing after %.0f s; listening to the caller again", OPENING_LINE_MAX_SEC)
    except Exception:
        logger.exception("opening line failed; listening to the caller again")
    finally:
        if not (ender is not None and ender.ending):
            session.input.set_audio_enabled(True)
            logger.info("caller audio resumed after the opening line")
