"""Transcript notes: things that happened during the call that people should see in the
transcript but the LLM never does (ignored filler words, blocked raw output). They go only into
the transcript sent to the backend, never into the conversation history."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone


def note_entry(kind: str, text: str, detail: str | None = None) -> dict:
    entry = {"role": "note", "text": text, "at": datetime.now(timezone.utc).isoformat(), "note": {"kind": kind}}
    if detail:
        entry["note"]["detail"] = detail
    return entry


class TranscriptNotesMixin:
    # Set by the entrypoint to append to the call's transcript
    log_transcript: Callable[[dict], None] | None = None

    def _note(self, kind: str, text: str, detail: str | None = None) -> None:
        if self.log_transcript:
            self.log_transcript(note_entry(kind, text, detail))
