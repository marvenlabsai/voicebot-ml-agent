"""Call recording: one stereo file per call, caller on the left channel and agent on the right.

Audio is captured from the session itself, after the callee answers: the caller's audio as it
arrives and the agent's speech where it actually played (an interrupted line stops where the
caller cut in). LiveKit's RecorderIO lines both up on one timeline and writes Ogg/Opus; with
RECORDING_FORMAT=wav (the default) that file is converted to 16 kHz 16-bit stereo WAV.

When the call ends the file is uploaded to the backend, which stores it in object storage.
Uploads that fail stay in RECORDINGS_DIR and are retried after later calls and when a worker
process starts. Anything older than RECORDING_KEEP_DAYS is deleted, uploaded or not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import wave
from pathlib import Path

import aiohttp
from livekit.agents import AgentSession
from livekit.agents.voice.recorder_io import RecorderIO

from reporter import AGENT_API_SECRET, BACKEND_URL

logger = logging.getLogger("voice-agent.recording")

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", str(Path(__file__).resolve().parent / ".cache" / "recordings")))
FORMAT = "ogg" if os.getenv("RECORDING_FORMAT", "wav").lower() == "ogg" else "wav"
WAV_SAMPLE_RATE = 16000
KEEP_DAYS = float(os.getenv("RECORDING_KEEP_DAYS", "7"))
CONTENT_TYPES = {"wav": "audio/wav", "ogg": "audio/ogg"}
UPLOAD_ATTEMPTS = 3
UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=300)
# A claim older than this belongs to a process that died mid-upload
STALE_CLAIM_SECONDS = 3600
# The backend refuses these for good (bad request, unknown call, too large, wrong type)
PERMANENT_STATUSES = {400, 401, 403, 404, 413, 415, 422}


class CallRecording:
    """Records one call. start() once the callee is on the line, finish() when it ends."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        self._recorder: RecorderIO | None = None
        self._raw = RECORDINGS_DIR / f"{call_id}.capture.ogg"

    async def start(self, session: AgentSession) -> None:
        if session.input.audio is None or session.output.audio is None:
            raise RuntimeError("the session has no audio input/output to record")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        recorder = RecorderIO(agent_session=session)
        session.input.audio = recorder.record_input(session.input.audio)
        session.output.audio = recorder.record_output(session.output.audio)
        await recorder.start(output_path=self._raw)
        self._recorder = recorder
        logger.info("recording call %s", self.call_id)

    async def finish(self) -> Path | None:
        """Stops recording and returns the finished file (with its .json sidecar), or None."""
        if self._recorder is None:
            return None
        recorder, self._recorder = self._recorder, None
        try:
            await recorder.aclose()
            return await asyncio.to_thread(_finalize, self.call_id, self._raw)
        except Exception:
            logger.exception("could not finish the recording of call %s", self.call_id)
            return None


def _finalize(call_id: str, raw: Path) -> Path | None:
    """Converts the capture to the configured format and writes the upload sidecar."""
    if not raw.exists() or raw.stat().st_size == 0:
        logger.warning("call %s: nothing was recorded", call_id)
        raw.unlink(missing_ok=True)
        return None
    out = RECORDINGS_DIR / f"{call_id}.{FORMAT}"
    if FORMAT == "wav":
        duration = _ogg_to_wav(raw, out)
        raw.unlink(missing_ok=True)
    else:
        duration = _ogg_duration(raw)
        os.replace(raw, out)
    _write_meta(out, {"callId": call_id, "format": FORMAT, "durationSec": round(duration, 2), "recordedAt": time.time()})
    return out


def _ogg_to_wav(src: Path, dst: Path) -> float:
    import av

    tmp = dst.with_suffix(".tmp")
    samples = 0
    resampler = av.AudioResampler(format="s16", layout="stereo", rate=WAV_SAMPLE_RATE)
    with av.open(str(src)) as container, wave.open(str(tmp), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(WAV_SAMPLE_RATE)

        def write(frames) -> None:
            nonlocal samples
            for f in frames:
                out.writeframes(f.to_ndarray().tobytes())
                samples += f.samples

        for frame in container.decode(audio=0):
            write(resampler.resample(frame))
        write(resampler.resample(None))
    os.replace(tmp, dst)
    return samples / WAV_SAMPLE_RATE


def _ogg_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        return float(container.duration / av.time_base) if container.duration else 0.0


def _meta_path(audio: Path) -> Path:
    return audio.with_suffix(".json")


def _write_meta(audio: Path, meta: dict) -> None:
    tmp = audio.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, _meta_path(audio))


async def _post(audio: Path, meta: dict) -> tuple[bool, bool]:
    """One upload attempt. Returns (done, permanent): done=True on success or permanent refusal."""
    url = f"{BACKEND_URL}/api/internal/calls/{meta['callId']}/recording"
    params = {"format": meta["format"]}
    if meta.get("durationSec"):
        params["durationSec"] = str(meta["durationSec"])
    headers = {
        "Authorization": f"Bearer {AGENT_API_SECRET}",
        "Content-Type": CONTENT_TYPES[meta["format"]],
        "Content-Length": str(audio.stat().st_size),
    }
    try:
        async with aiohttp.ClientSession(timeout=UPLOAD_TIMEOUT) as http:
            with audio.open("rb") as body:
                async with http.post(url, params=params, data=body, headers=headers) as resp:
                    if resp.status < 300:
                        return True, False
                    text = (await resp.text())[:300]
                    if resp.status in PERMANENT_STATUSES:
                        logger.error("backend refused recording of call %s: %s %s", meta["callId"], resp.status, text)
                        return True, True
                    logger.warning("recording upload of call %s failed: %s %s", meta["callId"], resp.status, text)
    except Exception as e:  # network errors, timeouts
        logger.warning("recording upload of call %s failed: %s", meta["callId"], e)
    return False, False


def _discard(audio: Path, meta_file: Path) -> None:
    audio.unlink(missing_ok=True)
    meta_file.unlink(missing_ok=True)


async def upload(audio: Path, *, attempts: int = UPLOAD_ATTEMPTS) -> bool:
    """Uploads a finished recording, retrying with backoff. The local copy is removed once the
    backend has it (or refuses it for good); otherwise it stays for a later retry."""
    if not AGENT_API_SECRET:
        logger.warning("AGENT_API_SECRET is not set: recording %s kept locally", audio.name)
        return False
    meta_file = _meta_path(audio)
    meta = json.loads(meta_file.read_text())
    for attempt in range(attempts):
        done, permanent = await _post(audio, meta)
        if done:
            _discard(audio, meta_file)
            if not permanent:
                logger.info("uploaded recording of call %s (%.0f s)", meta["callId"], meta.get("durationSec") or 0)
            return not permanent
        if attempt < attempts - 1:
            await asyncio.sleep(2 * 3**attempt)
    logger.warning("recording of call %s kept in %s for a later retry", meta["callId"], RECORDINGS_DIR)
    return False


async def flush_pending(*, budget_sec: float = 60) -> None:
    """Retries leftover uploads and deletes recordings past RECORDING_KEEP_DAYS.

    Several worker processes may run this at once: each file is claimed by renaming its sidecar.
    """
    if not RECORDINGS_DIR.exists():
        return
    deadline = time.monotonic() + budget_sec
    now = time.time()

    for claimed in RECORDINGS_DIR.glob("*.json.claimed-*"):
        try:
            if now - claimed.stat().st_mtime > STALE_CLAIM_SECONDS:
                os.replace(claimed, RECORDINGS_DIR / claimed.name.split(".claimed-")[0])
        except OSError:
            pass

    for path in RECORDINGS_DIR.iterdir():
        # Captures and conversions left behind by a crash, and sidecars without audio
        try:
            age_days = (now - path.stat().st_mtime) / 86400
        except OSError:
            continue
        if age_days > KEEP_DAYS:
            logger.info("deleting expired recording file %s", path.name)
            path.unlink(missing_ok=True)

    for meta_file in sorted(RECORDINGS_DIR.glob("*.json")):
        if time.monotonic() > deadline:
            break
        claim = meta_file.with_name(f"{meta_file.name}.claimed-{os.getpid()}")
        try:
            os.replace(meta_file, claim)
        except OSError:
            continue  # another process took it
        try:
            meta = json.loads(claim.read_text())
            audio = RECORDINGS_DIR / f"{meta['callId']}.{meta['format']}"
            if not audio.exists():
                claim.unlink(missing_ok=True)
                continue
            os.replace(claim, meta_file)  # upload() reads the sidecar from its usual name
            await upload(audio, attempts=1)
        except Exception:
            logger.exception("could not retry %s", meta_file.name)
            if claim.exists():
                os.replace(claim, meta_file)
