"""Filler-word filter: short acknowledgements ("haan", "ji", "hmm", "okay") don't cut the agent off.

While the agent's voice is playing, a transcript made only of the agent's filler words is
dropped before the session sees it. The framework then treats the caller's sound as a false
interruption and the agent carries on from where it paused. When the agent is silent the same
words pass through, since then "haan" is a real answer.

Matching ignores case and punctuation, and treats the two Devanagari nasal marks alike
(हाँ = हां), so the word list doesn't need every spelling variant. Combining vowel signs are
kept, so Hindi words aren't mangled when punctuation is stripped.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import AsyncIterable, Iterable

from livekit import rtc
from livekit.agents import Agent, stt
from livekit.agents.voice import ModelSettings

from notes import TranscriptNotesMixin

logger = logging.getLogger("voice-agent.filler")

# A transcript longer than this is real speech, even if every word is on the list
MAX_FILLER_WORDS = 4
CANDRABINDU, ANUSVARA = "ँ", "ं"
TRANSCRIPT_EVENTS = {
    stt.SpeechEventType.INTERIM_TRANSCRIPT,
    stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
    stt.SpeechEventType.FINAL_TRANSCRIPT,
}


def normalize(text: str) -> list[str]:
    """Lowercase words without punctuation or symbols; Devanagari nasal marks unified."""
    text = unicodedata.normalize("NFC", text).lower().replace(CANDRABINDU, ANUSVARA)
    # Category P (punctuation) and S (symbols) go; M (combining marks such as ा ी ं) stay
    kept = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return kept.split()


class FillerMatcher:
    def __init__(self, words: Iterable[str]):
        phrases = {tuple(normalize(w)) for w in words}
        # Longest phrases first, so "theek hai" is matched before "theek"
        self._phrases = sorted((p for p in phrases if p), key=len, reverse=True)

    def __bool__(self) -> bool:
        return bool(self._phrases)

    def is_filler(self, text: str) -> bool:
        """True if the text is nothing but filler words/phrases (and not too long)."""
        words = normalize(text)
        if not words or len(words) > MAX_FILLER_WORDS:
            return False
        i = 0
        while i < len(words):
            for phrase in self._phrases:
                if tuple(words[i : i + len(phrase)]) == phrase:
                    i += len(phrase)
                    break
            else:
                return False
        return True


class FillerFilterMixin(TranscriptNotesMixin):
    """Put before `Agent` in the bases. Call `set_filler_words()` before the session starts."""

    _filler: FillerMatcher | None = None
    _agent_audible = False
    _playback_watched = False

    def set_filler_words(self, words: Iterable[str] | None) -> None:
        matcher = FillerMatcher(words or [])
        self._filler = matcher if matcher else None

    async def on_enter(self) -> None:
        await super().on_enter()  # type: ignore[misc]
        self._watch_playback()

    def _watch_playback(self) -> None:
        """Tracks whether the caller is hearing the agent right now. A paused (possibly
        interrupted) line still counts: it only stops when playback finishes."""
        if self._playback_watched:
            return
        audio_out = self.session.output.audio  # type: ignore[attr-defined]
        if audio_out is None:
            return

        def started(_ev) -> None:
            self._agent_audible = True

        def finished(_ev) -> None:
            self._agent_audible = False

        audio_out.on("playback_started", started)
        audio_out.on("playback_finished", finished)
        self._playback_watched = True

    def _should_drop(self, ev: stt.SpeechEvent) -> bool:
        if self._filler is None or not self._agent_audible or ev.type not in TRANSCRIPT_EVENTS:
            return False
        text = ev.alternatives[0].text if ev.alternatives else ""
        if self._filler.is_filler(text):
            logger.info("filler while the agent speaks, ignored: %r", text)
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                # Shown in the transcript; never reaches the LLM (now or in later turns)
                self._note("filler", f"Caller said “{text.strip()}” while the agent was speaking (ignored)")
            return True
        return False

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        async for ev in Agent.default.stt_node(self, audio, model_settings):  # type: ignore[arg-type]
            if isinstance(ev, stt.SpeechEvent) and self._should_drop(ev):
                continue
            yield ev
