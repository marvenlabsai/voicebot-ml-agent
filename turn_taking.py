"""Turn-taking: when the caller's turn is over, when the caller may cut the agent off, and how
early the agent starts preparing its reply. Tuned for phone calls; each value can be overridden
from the environment.

- Voice activity detection notices speech after 50 ms and treats 250 ms of quiet as a pause,
  so turns end quickly.
- Endpointing waits min_delay..max_delay after the caller stops before treating the turn as
  over. The window depends on the speech-to-text provider (how fast it finalizes transcripts).
- An interruption needs at least 2 transcribed words and 0.5 s of speech. Anything shorter
  ("haan", a cough) pauses the agent, and if no real words follow within 1 s the agent resumes
  where it stopped.
- Preemptive generation starts the LLM reply, and its audio, while the caller is finishing
  their sentence, and throws it away if they keep talking. Off for scripted agents, whose
  matched turns don't use the LLM at all.
"""

from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# Per speech-to-text provider: Deepgram finalizes quickly, so a short window is enough
STT_ENDPOINTING: dict[str, dict] = {
    "deepgram": {"mode": "fixed", "min_delay": 0.2, "max_delay": 2.0},
}
DEFAULT_ENDPOINTING = {"mode": "fixed", "min_delay": 0.3, "max_delay": 2.5}


def vad_options() -> dict:
    """Silero VAD settings."""
    return {
        "min_speech_duration": _float("VAD_MIN_SPEECH_SECONDS", 0.05),
        "min_silence_duration": _float("VAD_MIN_SILENCE_SECONDS", 0.25),
        "activation_threshold": _float("VAD_ACTIVATION_THRESHOLD", 0.5),
    }


def endpointing(stt_provider: str) -> dict:
    opts = dict(STT_ENDPOINTING.get(stt_provider, DEFAULT_ENDPOINTING))
    if os.getenv("ENDPOINTING_MIN_SECONDS"):
        opts["min_delay"] = _float("ENDPOINTING_MIN_SECONDS", opts["min_delay"])
    if os.getenv("ENDPOINTING_MAX_SECONDS"):
        opts["max_delay"] = _float("ENDPOINTING_MAX_SECONDS", opts["max_delay"])
    return opts


def turn_handling(stt_provider: str, *, scripted: bool = False) -> dict:
    """The AgentSession `turn_handling` options for one call."""
    preemptive = os.getenv("PREEMPTIVE_GENERATION", "1") != "0" and not scripted
    return {
        "endpointing": endpointing(stt_provider),
        "interruption": {
            # Decided from voice activity + transcript words (not LiveKit's cloud model)
            "mode": "vad",
            "min_words": int(_float("INTERRUPTION_MIN_WORDS", 2)),
            "min_duration": _float("INTERRUPTION_MIN_SECONDS", 0.5),
            "false_interruption_timeout": _float("FALSE_INTERRUPTION_RESUME_SECONDS", 1.0),
            "resume_false_interruption": True,
            "discard_audio_if_uninterruptible": True,
        },
        "preemptive_generation": {
            "enabled": preemptive,
            "preemptive_tts": preemptive,
            "max_speech_duration": 10.0,
            "max_retries": 3,
        },
    }
