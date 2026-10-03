"""Worker capacity: how many calls one worker takes on, and how it starts and stops.

- A worker accepts calls until it runs MAX_JOBS_PER_WORKER of them, or its CPU is busier than
  LOAD_THRESHOLD; then LiveKit sends new calls to other workers.
- IDLE_PROCESSES job processes are kept started and prewarmed (VAD loaded), so a new call
  doesn't wait for a process to boot.
- On shutdown (SIGTERM, e.g. a deploy) the worker stops taking calls and gives running ones up
  to DRAIN_SECONDS to finish.
- LiveKit's worker serves a health check on HEALTH_PORT: GET / returns 200 "OK" (503 when it
  can't reach LiveKit), GET /worker returns active jobs and load as JSON.
"""

from __future__ import annotations

import os

from livekit.agents.worker import _DefaultLoadCalc


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_JOBS_PER_WORKER = max(1, _int("MAX_JOBS_PER_WORKER", 10))
IDLE_PROCESSES = max(0, _int("IDLE_PROCESSES", 3))
DRAIN_SECONDS = max(0, _int("DRAIN_SECONDS", 300))
HEALTH_PORT = _int("HEALTH_PORT", 8081)
# LiveKit requires a threshold below 1 in production
LOAD_THRESHOLD = min(0.99, max(0.1, float(os.getenv("LOAD_THRESHOLD", "0.9") or 0.9)))


def worker_load(worker) -> float:
    """Reported to LiveKit; at or above LOAD_THRESHOLD the worker gets no new calls.

    Jobs are scaled so that exactly MAX_JOBS_PER_WORKER running calls reach the threshold.
    """
    jobs = len(worker.active_jobs) / MAX_JOBS_PER_WORKER * LOAD_THRESHOLD
    try:
        cpu = _DefaultLoadCalc.get_load(worker)
    except Exception:
        cpu = 0.0
    return min(1.0, max(jobs, cpu))


def worker_options() -> dict:
    return {
        "load_fnc": worker_load,
        "load_threshold": LOAD_THRESHOLD,
        "num_idle_processes": IDLE_PROCESSES,
        "drain_timeout": DRAIN_SECONDS,
        "port": HEALTH_PORT,
    }
