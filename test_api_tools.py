"""External API tools: schema, argument checks, path placeholders, dedup, logging, safety."""

import asyncio
import json

import pytest
from aiohttp import web

import api_tools
from api_tools import ApiTool, build_api_tools


async def _serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _tool(base, log, **over):
    config = {
        "name": "get_order",
        "description": "Look up an order",
        "method": "GET",
        "url": f"{base}/orders/{{order_id}}",
        "headers": {"Authorization": "Bearer t0k"},
        "params": [
            {"name": "order_id", "type": "string", "required": True, "description": "Order number"},
            {"name": "count", "type": "integer", "required": False},
            {"name": "express", "type": "boolean"},
        ],
        "timeoutSec": 2,
        **over,
    }
    return ApiTool(config, log)


@pytest.fixture(autouse=True)
def allow_local(monkeypatch):
    monkeypatch.setattr(api_tools, "ALLOW_PRIVATE", True)


def test_schema():
    t = _tool("https://x.test", print)
    s = t.schema()
    assert s["name"] == "get_order"
    assert s["parameters"]["required"] == ["order_id"]
    assert s["parameters"]["properties"]["count"] == {"type": "integer"}
    assert s["description"].startswith("Look up an order")
    [ft] = build_api_tools([t.__dict__ | {"name": "get_order", "url": "https://x.test"}], print)
    assert ft.info.name == "get_order"


def test_get_fills_path_and_query_and_logs():
    seen, log = [], []

    async def handler(request):
        seen.append((request.method, request.path, dict(request.query), request.headers.get("Authorization")))
        return web.json_response({"status": "shipped"})

    async def run():
        runner, url = await _serve(handler)
        try:
            out = await _tool(url, log.append).run({"order_id": "A 1/2", "count": "3", "express": "yes", "junk": 1})
        finally:
            await runner.cleanup()
        return out

    out = asyncio.run(run())
    assert seen == [("GET", "/orders/A 1/2", {"count": "3", "express": "true"}, "Bearer t0k")]
    assert out.startswith("Success (HTTP 200)") and "shipped" in out
    [entry] = log
    assert entry["role"] == "tool" and entry["text"] == "get_order: HTTP 200"
    assert entry["tool"]["ok"] and entry["tool"]["status"] == 200 and "?" not in entry["tool"]["url"]
    assert json.loads(entry["tool"]["args"]) == {"order_id": "A 1/2", "count": 3, "express": True}


def test_post_sends_json_body():
    bodies = []

    async def handler(request):
        bodies.append(await request.json())
        return web.Response(status=201, text="created")

    async def run():
        runner, url = await _serve(handler)
        try:
            t = _tool(url, lambda e: None, method="POST", url=f"{url}/bookings", params=[{"name": "slot", "type": "string", "required": True}])
            return await t.run({"slot": "5pm"})
        finally:
            await runner.cleanup()

    assert asyncio.run(run()).startswith("Success (HTTP 201)")
    assert bodies == [{"slot": "5pm"}]


def test_missing_required_or_bad_type_is_not_sent():
    log = []
    t = _tool("https://never.test", log.append)
    out = asyncio.run(t.run({"count": 2}))
    assert "not sent" in out and "order_id" in out
    out = asyncio.run(t.run({"order_id": "1", "count": "two"}))
    assert '"count" must be a integer' in out
    assert [e["tool"]["ok"] for e in log] == [False, False]


def test_identical_request_within_window_is_skipped():
    hits, log = [], []

    async def handler(request):
        hits.append(1)
        return web.Response(text="ok")

    async def run():
        runner, url = await _serve(handler)
        try:
            t = _tool(url, log.append)
            first = await t.run({"order_id": "9"})
            second = await t.run({"order_id": "9"})
            other = await t.run({"order_id": "10"})
            return first, second, other
        finally:
            await runner.cleanup()

    first, second, other = asyncio.run(run())
    assert len(hits) == 2
    assert "do not repeat" in second and "ok" in second
    assert [e["tool"].get("skipped", False) for e in log] == [False, True, False]


def test_http_error_and_timeout_tell_llm_not_to_retry():
    async def handler(request):
        if "slow" in request.path:
            await asyncio.sleep(3)
        return web.Response(status=500, text="boom")

    async def run():
        runner, url = await _serve(handler)
        try:
            failed = await _tool(url, lambda e: None).run({"order_id": "1"})
            slow = await _tool(url, lambda e: None, url=f"{url}/slow/{{order_id}}", timeoutSec=1).run({"order_id": "1"})
            return failed, slow
        finally:
            await runner.cleanup()

    failed, slow = asyncio.run(run())
    assert "HTTP 500" in failed and "Do not retry" in failed
    assert "timed out" in slow and "Do not retry" in slow


def test_private_addresses_are_blocked(monkeypatch):
    monkeypatch.setattr(api_tools, "ALLOW_PRIVATE", False)
    log = []
    out = asyncio.run(_tool("http://127.0.0.1:9", log.append).run({"order_id": "1"}))
    assert "not a public address" in out and log[0]["tool"]["ok"] is False


def test_agent_gets_api_tools_after_end_call():
    import agent
    from end_call import CallEnder

    a = agent.build_agent(
        "p", "", {"tools": [{"name": "get_order", "description": "d", "url": "https://x.test/o", "params": []}]},
        session=None, ender=CallEnder(lambda r: None),
    )
    assert [t.info.name for t in a.tools] == ["end_call", "get_order"]
