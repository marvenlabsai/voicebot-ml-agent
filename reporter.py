"""Reports call progress and transcripts from the agent to our backend.

POST {BACKEND_URL}/api/internal/calls/{callId}/events
Authorization: Bearer {AGENT_API_SECRET}
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger("voice-agent.reporter")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001").rstrip("/")
AGENT_API_SECRET = os.getenv("AGENT_API_SECRET", "")


def iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else datetime.now().timestamp(), tz=timezone.utc).isoformat()


class CallReporter:
    def __init__(self, call_id: str | None):
        self.call_id = call_id
        self._http: aiohttp.ClientSession | None = None
        self._sent: set[str] = set()
        if call_id and not AGENT_API_SECRET:
            logger.warning("AGENT_API_SECRET is not set: call %s won't be reported to the backend", call_id)

    @property
    def enabled(self) -> bool:
        return bool(self.call_id and AGENT_API_SECRET)

    async def send(self, event_type: str, *, once: bool = True, **fields: Any) -> bool:
        """Sends one event. Retries transient failures; never raises."""
        if not self.enabled:
            return False
        if once and event_type in self._sent:
            return True
        self._sent.add(event_type)

        payload = {"type": event_type, "at": iso(), **{k: v for k, v in fields.items() if v is not None}}
        url = f"{BACKEND_URL}/api/internal/calls/{self.call_id}/events"
        headers = {"Authorization": f"Bearer {AGENT_API_SECRET}"}

        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

        for attempt in range(4):
            try:
                async with self._http.post(url, json=payload, headers=headers) as resp:
                    if resp.status < 300:
                        logger.info("reported %s for call %s", event_type, self.call_id)
                        return True
                    body = await resp.text()
                    # 4xx (bad secret, unknown call, validation) won't fix itself
                    if 400 <= resp.status < 500 and resp.status != 429:
                        logger.error("backend rejected %s for call %s: %s %s", event_type, self.call_id, resp.status, body[:300])
                        return False
                    logger.warning("backend error %s for %s: %s", resp.status, event_type, body[:200])
            except Exception as e:  # network errors, timeouts
                logger.warning("could not report %s for call %s (attempt %d): %s", event_type, self.call_id, attempt + 1, e)
            await asyncio.sleep(0.5 * 3**attempt)
        return False

    async def aclose(self) -> None:
        if self._http and not self._http.closed:
            await self._http.close()
