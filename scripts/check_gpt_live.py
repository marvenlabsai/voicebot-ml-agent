"""Checks that OPENAI_API_KEY can open a GPT-Live session, and prints what OpenAI says.

    python scripts/check_gpt_live.py [voice]

Run it where the agent runs (e.g. the Railway agent service's shell) so it uses the same key.
Prints the server's events for ~12 seconds: "session.started" means the key and model work;
an "error" event gives the reason (invalid key, no access to gpt-live-1, quota, ...).
"""

import asyncio
import json
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from livekit.plugins import openai  # noqa: E402


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    voice = sys.argv[1] if len(sys.argv) > 1 else "marin"
    backend = os.getenv("GPT_LIVE_BACKEND_MODEL")
    print(f"Opening a GPT-Live session (model gpt-live-1, voice {voice}, backend {backend or 'default'}) ...")
    async with aiohttp.ClientSession() as http:
        model = openai.realtime.GPTLiveModel(
            voice=voice,
            http_session=http,
            **({"responses_options": {"model": backend}} if backend else {}),
        )
        session = model.session()
        started = asyncio.Event()

        def on_event(ev: dict) -> None:
            kind = ev.get("type", "?")
            if kind == "error" or "error" in kind:
                print("ERROR from OpenAI:", json.dumps(ev, indent=2))
            elif kind == "session.started":
                print("OK: session.started — the key, model and voice work")
                started.set()
            else:
                print("event:", kind)

        session.on("openai_server_event_received", on_event)
        session.on("error", lambda e: print("session error:", getattr(e, "error", e)))
        # The plugin only starts the session once it's configured (as the agent framework does)
        await session._update_session(instructions="You are a test assistant. Say hello in one short sentence.")
        try:
            await asyncio.wait_for(started.wait(), timeout=12)
        except asyncio.TimeoutError:
            print("No session.started within 12 s (see any ERROR above)")
        await session.aclose()


asyncio.run(main())
