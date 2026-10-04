"""Usage per call, sent to the backend with the `ended` report.

Totals come from the session's own usage summary (one entry per model): LLM tokens, TTS
characters and seconds of audio, STT seconds of audio. Each model used is listed too, so costs
can be worked out per provider. Billed telephony minutes are computed by the backend from the
call's duration. Scripted agents also report their pre-synthesized line cache.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentSession

logger = logging.getLogger("voice-agent.usage")


def _r(x: float) -> float:
    return round(float(x or 0), 2)


def usage_report(session: AgentSession, agent: object, transcript: list[dict]) -> dict | None:
    try:
        model_usage = session.usage.model_usage
    except Exception:
        logger.exception("could not read session usage")
        return None

    llm = {"inputTokens": 0, "cachedInputTokens": 0, "outputTokens": 0}
    tts = {"characters": 0, "audioSec": 0.0}
    stt = {"audioSec": 0.0}
    realtime_sec = 0.0  # realtime (speech-to-speech) models are billed by session time
    models = []
    for u in model_usage:
        kind = getattr(u, "type", "")
        base = {"provider": getattr(u, "provider", "") or "", "model": getattr(u, "model", "") or ""}
        if kind == "llm_usage":
            llm["inputTokens"] += u.input_tokens
            llm["cachedInputTokens"] += u.input_cached_tokens
            llm["outputTokens"] += u.output_tokens
            session = getattr(u, "session_duration", 0.0) or 0.0
            realtime_sec += session
            entry = {"kind": "llm", **base, "inputTokens": u.input_tokens, "cachedInputTokens": u.input_cached_tokens, "outputTokens": u.output_tokens}
            if session:
                entry["sessionSec"] = _r(session)
            models.append(entry)
        elif kind == "tts_usage":
            tts["characters"] += u.characters_count
            tts["audioSec"] += u.audio_duration
            models.append({"kind": "tts", **base, "characters": u.characters_count, "audioSec": _r(u.audio_duration)})
        elif kind == "stt_usage":
            stt["audioSec"] += u.audio_duration
            models.append({"kind": "stt", **base, "audioSec": _r(u.audio_duration)})
    tts["audioSec"] = _r(tts["audioSec"])
    stt["audioSec"] = _r(stt["audioSec"])

    report: dict = {
        "llm": llm,
        "tts": tts,
        "stt": stt,
        "realtimeSec": _r(realtime_sec),
        "models": models,
        "toolCalls": sum(1 for t in transcript if t.get("role") == "tool" and not t.get("tool", {}).get("skipped") and t.get("tool", {}).get("ms") is not None),
    }
    audio = getattr(agent, "_audio", None)  # scripted agents: their pre-synthesized line cache
    if audio is not None and isinstance(getattr(audio, "stats", None), dict):
        report["scriptCache"] = dict(audio.stats)
    return report
