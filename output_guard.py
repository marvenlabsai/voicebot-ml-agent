"""Output guard: the agent never speaks raw machine output, and text reads cleanly for TTS.

The LLM's text is checked as it streams, before it reaches TTS, the live transcript or the
conversation history. It's released a sentence at a time; a sentence is held back if it shows:

- a JSON object or an angle-bracket tag (`{"status": …}`, `<function=end_call>`)
- tool-call syntax (`to=functions.x`, `functions.x(`)
- an echoed tool result (`SUCCESS:`, `ERROR:`, `Success (HTTP 200)`, `"key": value`)
- a stack trace or exception text, or a code fence

On a hit the rest of that reply is dropped (nothing of it is spoken or stored), the hit is noted
in the call transcript, and the LLM is asked for the reply again with a one-off instruction that
isn't kept in the history. If the second reply also contains raw output, the offending parts
are cut out instead of asking a third time.

Released text is also tidied for speech: line breaks become spaces, markdown symbols go, and a
missing space after a full stop ("know.Kya") is added so the TTS doesn't read the dot out.
"""

from __future__ import annotations

import logging
import re

from livekit.agents import Agent, llm

from notes import TranscriptNotesMixin

logger = logging.getLogger("voice-agent.output-guard")

SENTENCE_END = set(".!?।…\n")
# Release a long run without sentence punctuation anyway, after checking it
MAX_HOLD_CHARS = 300
SNIPPET_CHARS = 200

RAW_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tool call syntax", re.compile(r"to=functions\.|\bfunctions\.\w+\s*\(")),
    ("tool result", re.compile(r"\b(?:SUCCESS|ERROR)\s*:|\bSuccess \(HTTP \d{3}\)|\bHTTP \d{3}\)?\.? Response:")),
    ("JSON", re.compile(r'"[\w-]+"\s*:\s*(?:"|\d|\[|\{|true\b|false\b|null\b)')),
    (
        "stack trace",
        re.compile(r'Traceback \(most recent call last\)|File "[^"]+", line \d+|\bat [\w.$<>]+\([\w/\\.:-]+:\d+:\d+\)'),
    ),
    ("error text", re.compile(r"\b[A-Z]\w*(?:Exception|Error):\s+\S")),
    ("code", re.compile(r"```")),
]
OPENERS = {"{": ("}", "JSON"), "<": (">", "tag")}

# Letters (Latin, Devanagari incl. vowel signs) on both sides of . ! ? or ।
_LETTER = r"A-Za-zÀ-ɏऀ-ॿ"
_MISSING_SPACE = re.compile(rf"(?<=[{_LETTER}])([.!?।])(?=[{_LETTER}])")
_MARKDOWN = re.compile(r"[*#`]+|^\s*[-•]\s+", re.MULTILINE)
_SPACES = re.compile(r"[ \t]{2,}")
_SPEAKABLE = re.compile(rf"[0-9{_LETTER}]")

RETRY_INSTRUCTIONS = (
    "Your previous reply was not spoken because it contained raw data, code or function-call text. "
    "Answer again in plain, natural spoken language only: no JSON, braces, tags, code, function names, "
    "or raw tool output. If an action was completed, briefly confirm it in your own words."
)


def tidy(text: str) -> str:
    """Text as it should reach the TTS."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = _MARKDOWN.sub("", text)
    text = _MISSING_SPACE.sub(r"\1 ", text)
    text = _SPACES.sub(" ", text)
    if text.rstrip() and text.rstrip()[-1] in ".!?।…:" and not text.endswith(" "):
        text += " "  # so the next sentence isn't glued on ("know." + "Kya")
    return text


class TextGuard:
    """Streaming filter for one LLM reply.

    strict: stop at the first raw output and report it (`hit`); otherwise cut raw parts out
    and keep going (`removed` lists what was cut).
    """

    def __init__(self, *, strict: bool):
        self.strict = strict
        self.hit: tuple[str, str] | None = None  # (kind, snippet)
        self.removed: list[str] = []
        self._buf = ""
        self._block = ""
        self._closer = ""
        self._depth = 0
        self._block_kind = ""
        self._ends_in_space = True  # nothing released yet: drop leading spaces

    def _flagged(self, text: str) -> str | None:
        for kind, pattern in RAW_PATTERNS:
            if pattern.search(text):
                return kind
        return None

    def _report(self, kind: str, snippet: str) -> None:
        if self.strict:
            self.hit = (kind, snippet.strip()[:SNIPPET_CHARS])
        else:
            self.removed.append(snippet.strip()[:SNIPPET_CHARS])

    def _emit(self, out: list[str], text: str) -> None:
        if self._ends_in_space:
            text = text.lstrip()
        if text:
            out.append(text)
            self._ends_in_space = text[-1].isspace()

    def _release(self, out: list[str]) -> None:
        text, self._buf = self._buf, ""
        if not text.strip():
            if text:
                self._emit(out, " ")
            return
        kind = self._flagged(text)
        if kind:
            self._report(kind, text)
            return
        clean = tidy(text)
        if _SPEAKABLE.search(clean):
            self._emit(out, clean)

    def feed(self, text: str) -> list[str]:
        """Clean text that's safe to speak now. Stops producing once `hit` is set."""
        out: list[str] = []
        for i, ch in enumerate(text):
            if self.hit:
                self.extend_snippet(text[i:])
                break
            if self._depth:
                self._block += ch
                if ch == self._block[0]:
                    self._depth += 1
                elif ch == self._closer:
                    self._depth -= 1
                    if not self._depth:
                        self._report(self._block_kind, self._block)
                        self._block = ""
                continue
            if ch in OPENERS:
                if self.strict:
                    # Whatever led up to it ("Here is the result:") goes too
                    self._report(OPENERS[ch][1], self._buf + ch)
                    self._buf = ""
                    self.extend_snippet(text[i + 1 :])
                    break
                self._release(out)
                self._closer, self._block_kind = OPENERS[ch]
                self._block, self._depth = ch, 1
                continue
            self._buf += ch
            if ch in SENTENCE_END or len(self._buf) >= MAX_HOLD_CHARS:
                self._release(out)
        return out

    def finish(self) -> list[str]:
        """The reply ended: release what's left (an unclosed block counts as raw output)."""
        out: list[str] = []
        if self.hit:
            return out
        if self._depth:
            self._report(self._block_kind, self._block)
            self._block, self._depth = "", 0
        if not self.hit:
            self._release(out)
        return out

    def extend_snippet(self, text: str) -> bool:
        """After a hit: collect a bit more of what the LLM was writing, for the note."""
        if not self.hit:
            return False
        kind, snippet = self.hit
        if len(snippet) >= SNIPPET_CHARS:
            return False
        self.hit = (kind, (snippet + text)[:SNIPPET_CHARS])
        return len(self.hit[1]) < SNIPPET_CHARS


def _without_text(chunk: llm.ChatChunk) -> llm.ChatChunk:
    return chunk.model_copy(update={"delta": chunk.delta.model_copy(update={"content": None})})


class OutputGuardMixin(TranscriptNotesMixin):
    """Put before `Agent` in the bases."""

    _regenerating = False  # the next reply is the one asked for after a hit

    async def llm_node(self, chat_ctx, tools, model_settings):
        guard = TextGuard(strict=not self._regenerating)
        self._regenerating = False
        source = Agent.default.llm_node(self, chat_ctx, tools, model_settings)  # type: ignore[arg-type]
        async for chunk in source:
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                text = chunk.delta.content
                if chunk.delta.tool_calls and not guard.hit:
                    yield _without_text(chunk)
            else:
                if not guard.hit:
                    yield chunk  # tool calls, usage, flushes
                continue
            if guard.hit:
                if not guard.extend_snippet(text):
                    break
                continue
            for piece in guard.feed(text):
                yield piece
        else:
            for piece in guard.finish():
                yield piece

        if guard.hit:
            self._regenerate(*guard.hit)
        elif guard.removed:
            logger.warning("cut raw output from a regenerated reply: %r", guard.removed)
            self._note("output_guard", "Raw output removed from the reply", "; ".join(guard.removed)[:SNIPPET_CHARS])

    def _regenerate(self, kind: str, snippet: str) -> None:
        logger.warning("raw output blocked (%s): %r; asking the LLM again", kind, snippet)
        self._note("output_guard", f"{kind} detected in the reply, not spoken; reply regenerated", snippet)
        self._regenerating = True
        try:
            # The instruction applies to this one reply only; it isn't added to the history
            self.session.generate_reply(instructions=RETRY_INSTRUCTIONS)  # type: ignore[attr-defined]
        except Exception:
            self._regenerating = False
            logger.exception("could not regenerate the reply")
