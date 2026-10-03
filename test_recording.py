"""Call recordings: Ogg → stereo WAV, upload with retries, leftovers retried, old files expired."""

import asyncio
import json
import os
import time
import wave

import numpy as np
import pytest
from aiohttp import web

import recording


def _write_ogg(path, seconds=1.5, rate=48000):
    """A stereo Ogg/Opus file: a tone on the left (caller), silence on the right (agent)."""
    import av

    t = np.arange(int(seconds * rate)) / rate
    left = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    block = np.stack([left, np.zeros_like(left)])
    with av.open(str(path), "w", format="ogg") as c:
        s = c.add_stream("libopus", rate=rate, layout="stereo")
        for i in range(0, block.shape[1], 960):
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(block[:, i : i + 960]), format="fltp", layout="stereo")
            f.sample_rate = rate
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)


@pytest.fixture
def rec_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(recording, "AGENT_API_SECRET", "s3cret")
    return tmp_path


def _finished(rec_dir, call_id="c1", fmt="wav"):
    raw = rec_dir / f"{call_id}.capture.ogg"
    _write_ogg(raw)
    recording.FORMAT = fmt
    try:
        return recording._finalize(call_id, raw)
    finally:
        recording.FORMAT = "wav"


def test_capture_becomes_stereo_wav_with_sidecar(rec_dir):
    out = _finished(rec_dir)
    assert out == rec_dir / "c1.wav" and not (rec_dir / "c1.capture.ogg").exists()
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, 16000)
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
    assert np.abs(data[:, 0]).mean() > 1000 > np.abs(data[:, 1]).mean()  # caller left, agent right
    meta = json.loads((rec_dir / "c1.json").read_text())
    assert meta["callId"] == "c1" and meta["format"] == "wav" and 1.3 < meta["durationSec"] < 1.7


def test_ogg_format_keeps_the_capture(rec_dir):
    out = _finished(rec_dir, fmt="ogg")
    assert out == rec_dir / "c1.ogg"
    assert json.loads((rec_dir / "c1.json").read_text())["format"] == "ogg"


def test_empty_capture_gives_nothing(rec_dir):
    raw = rec_dir / "c2.capture.ogg"
    raw.write_bytes(b"")
    assert recording._finalize("c2", raw) is None and not raw.exists()


async def _backend(statuses, received):
    """A stand-in backend that answers uploads with the given statuses in turn."""

    async def handler(request):
        received.append((request.match_info["id"], dict(request.query), request.headers["Content-Type"], request.headers["Authorization"], len(await request.read())))
        return web.Response(status=statuses.pop(0) if statuses else 201)

    app = web.Application()
    app.router.add_post("/api/internal/calls/{id}/recording", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def _run_with_backend(monkeypatch, statuses, fn):
    received = []

    async def run():
        runner, url = await _backend(statuses, received)
        monkeypatch.setattr(recording, "BACKEND_URL", url)
        try:
            return await fn()
        finally:
            await runner.cleanup()

    return asyncio.run(run()), received


def test_upload_sends_file_and_removes_local_copy(rec_dir, monkeypatch):
    out = _finished(rec_dir)
    size = out.stat().st_size
    ok, received = _run_with_backend(monkeypatch, [201], lambda: recording.upload(out))
    assert ok and not out.exists() and not (rec_dir / "c1.json").exists()
    [(call_id, query, ctype, auth, length)] = received
    assert (call_id, query["format"], ctype, auth, length) == ("c1", "wav", "audio/wav", "Bearer s3cret", size)
    assert 1.3 < float(query["durationSec"]) < 1.7


def test_upload_retries_then_keeps_file(rec_dir, monkeypatch):
    monkeypatch.setattr(recording.asyncio, "sleep", _no_sleep)
    out = _finished(rec_dir)
    ok, received = _run_with_backend(monkeypatch, [503, 502, 500], lambda: recording.upload(out))
    assert not ok and len(received) == 3 and out.exists()  # left for a later retry


def test_upload_retry_succeeds(rec_dir, monkeypatch):
    monkeypatch.setattr(recording.asyncio, "sleep", _no_sleep)
    out = _finished(rec_dir)
    ok, received = _run_with_backend(monkeypatch, [503, 201], lambda: recording.upload(out))
    assert ok and len(received) == 2 and not out.exists()


def test_permanent_refusal_drops_file(rec_dir, monkeypatch):
    out = _finished(rec_dir)
    ok, received = _run_with_backend(monkeypatch, [404], lambda: recording.upload(out))
    assert not ok and len(received) == 1 and not out.exists()


def test_flush_pending_uploads_leftovers_and_expires_old_files(rec_dir, monkeypatch):
    kept = _finished(rec_dir, "kept")
    old = _finished(rec_dir, "old")
    week_ago = time.time() - 8 * 86400
    for p in (old, rec_dir / "old.json"):
        os.utime(p, (week_ago, week_ago))
    stale_claim = rec_dir / "stuck.json.claimed-999"
    _finished(rec_dir, "stuck")
    os.replace(rec_dir / "stuck.json", stale_claim)
    os.utime(stale_claim, (time.time() - 7200,) * 2)

    _, received = _run_with_backend(monkeypatch, [], lambda: recording.flush_pending())
    assert sorted(r[0] for r in received) == ["kept", "stuck"]  # the expired one was deleted, not sent
    assert list(rec_dir.iterdir()) == []
    assert not kept.exists()


async def _no_sleep(_s):
    return None


def test_live_session_audio_is_recorded_in_stereo(rec_dir):
    """The real recorder, wrapped around a session's audio: caller left, agent right, in time."""
    from types import SimpleNamespace

    from livekit import rtc
    from livekit.agents.voice import io

    def tone(freq, n, rate):
        t = np.arange(n) / rate
        return (8000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)

    class CallerAudio(io.AudioInput):
        """0.6 s of caller speech, delivered in real time."""

        def __init__(self):
            super().__init__(label="caller")
            self.sent = 0

        async def __anext__(self):
            if self.sent >= 30:
                raise StopAsyncIteration
            await asyncio.sleep(0.02)
            self.sent += 1
            return rtc.AudioFrame(tone(300, 960, 48000).tobytes(), 48000, 1, 960)

    class Speaker(io.AudioOutput):
        """Plays agent audio instantly and reports it, like the room's audio sink."""

        def __init__(self):
            super().__init__(label="speaker", capabilities=io.AudioOutputCapabilities(pause=False))
            self.played = 0.0

        async def capture_frame(self, frame):
            await super().capture_frame(frame)
            if self.played == 0:
                self.on_playback_started(created_at=time.time())
            self.played += frame.duration

        def flush(self):
            super().flush()
            self.on_playback_finished(playback_position=self.played, interrupted=False)

        def clear_buffer(self):
            pass

    session = SimpleNamespace(input=SimpleNamespace(audio=CallerAudio()), output=SimpleNamespace(audio=Speaker()))

    async def run():
        rec = recording.CallRecording("live")
        await rec.start(session)
        async for _ in session.input.audio:  # the session reading the caller
            pass
        out = session.output.audio
        for _ in range(25):  # 0.5 s of agent speech at 24 kHz
            await out.capture_frame(rtc.AudioFrame(tone(800, 480, 24000).tobytes(), 24000, 1, 480))
        out.flush()
        await asyncio.sleep(0.6)
        return await rec.finish()

    path = asyncio.run(run())
    assert path == rec_dir / "live.wav"
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2).astype(float)
    loud = lambda ch: np.flatnonzero(np.abs(data[:, ch]) > 2000) / 16000  # noqa: E731
    caller, agent = loud(0), loud(1)
    assert 0.4 < caller[-1] - caller[0] < 1.3  # the caller (placed by arrival time), on the left
    assert 0.3 < agent[-1] - agent[0] < 0.7  # ~0.5 s of agent, on the right
    assert agent[0] >= caller[-1] - 0.05  # the agent spoke after the caller, not over them
