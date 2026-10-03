"""Worker capacity and the per-call time limit."""

import asyncio
from types import SimpleNamespace

import capacity
from end_call import CallEnder, hang_up_after_goodbye


def test_jobs_reach_the_threshold_exactly_at_the_limit(monkeypatch):
    monkeypatch.setattr(capacity._DefaultLoadCalc, "get_load", classmethod(lambda cls, w: 0.0))
    load = lambda n: capacity.worker_load(SimpleNamespace(active_jobs=[0] * n))  # noqa: E731
    limit = capacity.MAX_JOBS_PER_WORKER
    assert load(limit - 1) < capacity.LOAD_THRESHOLD <= load(limit)


def test_busy_cpu_also_makes_the_worker_full(monkeypatch):
    monkeypatch.setattr(capacity._DefaultLoadCalc, "get_load", classmethod(lambda cls, w: 0.95))
    assert capacity.worker_load(SimpleNamespace(active_jobs=[])) >= capacity.LOAD_THRESHOLD


def test_worker_options_are_valid_for_livekit():
    from livekit.agents import WorkerOptions

    o = WorkerOptions(entrypoint_fnc=lambda ctx: None, **capacity.worker_options())
    assert o.load_threshold < 1 and o.drain_timeout == 300 and o.num_idle_processes == 3 and o.port == 8081


def test_call_limit_is_capped_at_ten_minutes():
    import agent

    assert agent.call_limit_seconds({}) == 600
    assert agent.call_limit_seconds({"maxCallSec": 180}) == 180
    assert agent.call_limit_seconds({"maxCallSec": 3600}) == 600


class _Handle:
    def __init__(self):
        self.callbacks = []

    def add_done_callback(self, cb):
        self.callbacks.append(cb)


def test_goodbye_then_hang_up():
    ended = []
    said = []

    def say(text, allow_interruptions=True):
        said.append((text, allow_interruptions))
        return handle

    handle = _Handle()

    async def run():
        ender = CallEnder(ended.append)
        hang_up_after_goodbye(SimpleNamespace(say=say), ender, "Thanks, bye!", "Max call duration reached (10 min)")
        assert ender.ending and ended == []
        handle.callbacks[0](handle)

    asyncio.run(run())
    assert said == [("Thanks, bye!", False)] and ended == ["Max call duration reached (10 min)"]


def test_no_goodbye_hangs_up_at_once():
    ended = []
    hang_up_after_goodbye(None, CallEnder(ended.append), "", "Max call duration reached (5 min)")
    assert ended == ["Max call duration reached (5 min)"]
