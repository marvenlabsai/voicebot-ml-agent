"""External API tools: HTTP endpoints configured on the agent that the LLM can call mid-call.

The backend sends each tool with the call's {variables} already filled in. The URL may still
contain {placeholders} named after the tool's own parameters; those are filled with what the
LLM passes. Other parameters go in the query string (GET/DELETE) or the JSON body.

Every request (and every request that wasn't sent) is logged into the call transcript.
Identical requests within DEDUP_SECONDS reuse the earlier result instead of calling again.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp
from livekit.agents import RunContext, function_tool

logger = logging.getLogger("voice-agent.api-tools")

DEDUP_SECONDS = 15
# What the LLM sees of a response, and what the transcript keeps
MAX_RESULT_CHARS = 3000
MAX_LOG_CHARS = 1000
MAX_ARGS_LOG_CHARS = 1000
# Tools may not reach private/internal addresses unless explicitly allowed (local development)
ALLOW_PRIVATE = os.getenv("API_TOOLS_ALLOW_PRIVATE") == "1"

JSON_TYPES = {"string": "string", "number": "number", "integer": "integer", "boolean": "boolean"}
PLACEHOLDER_RE = re.compile(r"\{\s*([A-Za-z_][A-Za-z0-9_]{0,63})\s*\}")

TOOL_GUIDE = (
    "\n\nWhile it runs, the caller hears nothing, so say a few words first (like \"one moment, let me check\")."
    " Use the result in plain spoken language; never read out raw data, codes or JSON."
)


class ToolInputError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _coerce(value: Any, kind: str, name: str) -> Any:
    """The LLM's value as the parameter's type."""
    try:
        if kind == "integer":
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
            if not number.is_integer():
                raise ValueError
            return int(number)
        if kind == "number":
            if isinstance(value, bool):
                raise ValueError
            return float(value) if not isinstance(value, int) else value
        if kind == "boolean":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "yes", "1"):
                return True
            if text in ("false", "no", "0"):
                return False
            raise ValueError
        return value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    except (TypeError, ValueError):
        raise ToolInputError(f'"{name}" must be a {kind}') from None


async def _check_public(url: str) -> None:
    """Refuses URLs whose host resolves to a private, loopback or link-local address."""
    if ALLOW_PRIVATE:
        return
    host = urlsplit(url).hostname
    if not host:
        raise ToolInputError("the URL has no host")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ToolInputError(f"could not resolve {host}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ToolInputError(f"{host} is not a public address")


def _query_value(value: Any) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


class ApiTool:
    """One configured endpoint, as an LLM tool."""

    def __init__(self, config: dict, log: Callable[[dict], None]):
        self.name: str = config["name"]
        self.method: str = (config.get("method") or "GET").upper()
        self.url: str = config["url"]
        self.headers: dict[str, str] = {k: str(v) for k, v in (config.get("headers") or {}).items() if v != ""}
        self.params: list[dict] = config.get("params") or []
        self.timeout = float(config.get("timeoutSec") or 10)
        self._description = (config.get("description") or "").strip()
        self._log = log
        self._recent: dict[str, tuple[float, str]] = {}  # request key -> (when, result for the LLM)

    def schema(self) -> dict:
        properties = {}
        for p in self.params:
            prop: dict[str, Any] = {"type": JSON_TYPES.get(p.get("type"), "string")}
            if p.get("description"):
                prop["description"] = p["description"]
            properties[p["name"]] = prop
        return {
            "name": self.name,
            "description": self._description + TOOL_GUIDE,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [p["name"] for p in self.params if p.get("required")],
            },
        }

    def as_function_tool(self):
        async def call_api(raw_arguments: dict[str, object], context: RunContext) -> str:
            return await self.run(raw_arguments)

        return function_tool(call_api, raw_schema=self.schema())

    def _arguments(self, raw: dict | None) -> dict[str, Any]:
        """Known parameters with values, typed; raises ToolInputError if a required one is missing."""
        raw = raw or {}
        args: dict[str, Any] = {}
        for p in self.params:
            value = raw.get(p["name"])
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            args[p["name"]] = _coerce(value, p.get("type") or "string", p["name"])
        missing = [p["name"] for p in self.params if p.get("required") and p["name"] not in args]
        if missing:
            raise ToolInputError(f"missing {', '.join(missing)}")
        return args

    def _request(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The URL with path placeholders filled, and the arguments left for the query/body."""
        rest = dict(args)

        def fill(m: re.Match) -> str:
            name = m.group(1)
            if name in rest:
                return quote(_query_value(rest.pop(name)), safe="")
            return m.group(0)

        url = PLACEHOLDER_RE.sub(fill, self.url)
        if PLACEHOLDER_RE.search(url):
            raise ToolInputError(f"no value for {', '.join(PLACEHOLDER_RE.findall(url))} in the URL")
        return url, rest

    def _entry(self, *, outcome: str, ok: bool, args: dict, url: str = "", status: int | None = None,
               ms: int | None = None, response: str = "", skipped: bool = False) -> dict:
        tool = {
            "name": self.name,
            "method": self.method,
            "url": _short(url.split("?")[0], 2000) if url else "",
            "args": _short(json.dumps(args, ensure_ascii=False, default=str), MAX_ARGS_LOG_CHARS),
            "ok": ok,
        }
        if status is not None:
            tool["status"] = status
        if ms is not None:
            tool["ms"] = ms
        if response:
            tool["response"] = _short(response, MAX_LOG_CHARS)
        if skipped:
            tool["skipped"] = True
        return {"role": "tool", "text": f"{self.name}: {outcome}", "at": _now(), "tool": tool}

    async def run(self, raw: dict | None) -> str:
        try:
            args = self._arguments(raw)
            url, rest = self._request(args)
        except ToolInputError as e:
            self._log(self._entry(outcome=f"not sent ({e})", ok=False, args=dict(raw or {})))
            return f"The request was not sent: {e}. Ask the caller for what's missing, then try again."

        key = json.dumps(args, sort_keys=True, default=str)
        now = time.monotonic()
        self._recent = {k: v for k, v in self._recent.items() if now - v[0] < DEDUP_SECONDS}
        if key in self._recent:
            ago, result = now - self._recent[key][0], self._recent[key][1]
            self._log(self._entry(outcome="duplicate skipped", ok=True, skipped=True, args=args, url=url))
            return f"You made this exact request {ago:.0f} seconds ago; do not repeat it. Its result was: {result}"

        result = await self._send(url, rest, args)
        self._recent[key] = (time.monotonic(), result)
        return result

    async def _send(self, url: str, rest: dict[str, Any], args: dict[str, Any]) -> str:
        started = time.perf_counter()
        status: int | None = None
        try:
            await _check_public(url)
            kwargs: dict[str, Any] = {"headers": self.headers, "allow_redirects": False}
            if self.method in ("GET", "DELETE"):
                kwargs["params"] = {k: _query_value(v) for k, v in rest.items()}
            else:
                kwargs["json"] = rest
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.request(self.method, url, **kwargs) as resp:
                    status = resp.status
                    body = await resp.text(errors="replace")
        except ToolInputError as e:
            return self._failed(url, args, started, f"blocked: {e}", status)
        except asyncio.TimeoutError:
            return self._failed(url, args, started, f"timed out after {self.timeout:.0f}s", status)
        except aiohttp.ClientError as e:
            return self._failed(url, args, started, f"connection error: {e}", status)

        ms = round((time.perf_counter() - started) * 1000)
        ok = 200 <= status < 300
        body = body.strip()
        self._log(self._entry(outcome=f"HTTP {status}", ok=ok, args=args, url=url, status=status, ms=ms, response=body))
        logger.info("tool %s → %s in %d ms", self.name, status, ms)
        if ok:
            return f"Success (HTTP {status}). Response: {_short(body, MAX_RESULT_CHARS) or '(empty)'}"
        return (
            f"The request failed (HTTP {status}). Response: {_short(body, MAX_RESULT_CHARS) or '(empty)'}. "
            "Tell the caller briefly that this couldn't be done right now. Do not retry the same request."
        )

    def _failed(self, url: str, args: dict, started: float, why: str, status: int | None) -> str:
        ms = round((time.perf_counter() - started) * 1000)
        logger.warning("tool %s failed: %s", self.name, why)
        self._log(self._entry(outcome=f"failed ({why})", ok=False, args=args, url=url, status=status, ms=ms))
        return (
            f"The request could not be completed ({why}). Tell the caller briefly that this couldn't be done "
            "right now. Do not retry the same request."
        )


def build_api_tools(configs: list | None, log: Callable[[dict], None]) -> list:
    """LLM tools for the agent's configured endpoints. A broken config skips that tool only."""
    tools = []
    for config in configs or []:
        try:
            tools.append(ApiTool(config, log).as_function_tool())
        except Exception:
            logger.exception("could not set up API tool %r", (config or {}).get("name"))
    if tools:
        logger.info("%d API tool(s) available", len(tools))
    return tools
