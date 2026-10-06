"""Usage report sent with the ended event."""

from types import SimpleNamespace

from livekit.agents.metrics.usage import LLMModelUsage, STTModelUsage, TTSModelUsage

from usage import usage_report


def _session(*usage):
    return SimpleNamespace(usage=SimpleNamespace(model_usage=list(usage)))


def test_totals_and_models():
    s = _session(
        LLMModelUsage(provider="cerebras", model="gpt-oss-120b", input_tokens=1200, input_cached_tokens=800, output_tokens=150),
        TTSModelUsage(provider="cartesia", model="sonic-2", characters_count=420, audio_duration=31.237),
        STTModelUsage(provider="deepgram", model="nova-3", audio_duration=58.5),
    )
    transcript = [
        {"role": "user", "text": "hi"},
        {"role": "tool", "text": "x: HTTP 200", "tool": {"ms": 120}},
        {"role": "tool", "text": "x: duplicate skipped", "tool": {"skipped": True}},
        {"role": "tool", "text": "x: not sent", "tool": {}},
    ]
    r = usage_report(s, SimpleNamespace(), transcript)
    assert r["llm"] == {"inputTokens": 1200, "cachedInputTokens": 800, "outputTokens": 150}
    assert r["tts"] == {"characters": 420, "audioSec": 31.24} and r["stt"] == {"audioSec": 58.5}
    assert [m["kind"] for m in r["models"]] == ["llm", "tts", "stt"]
    assert r["models"][0]["model"] == "gpt-oss-120b" and r["toolCalls"] == 1
    assert "scriptCache" not in r


def test_realtime_models_report_audio_tokens_and_session_time():
    s = _session(
        LLMModelUsage(provider="google", model="gemini-live", input_tokens=900, output_tokens=300, input_audio_tokens=700, output_audio_tokens=250),
        LLMModelUsage(provider="openai", model="gpt-live-1", input_tokens=50, output_tokens=20, session_duration=61.234),
    )
    r = usage_report(s, SimpleNamespace(), [])
    assert r["llm"]["inputAudioTokens"] == 700 and r["llm"]["outputAudioTokens"] == 250
    gemini, gpt = r["models"]
    assert gemini["inputAudioTokens"] == 700 and gemini["outputAudioTokens"] == 250 and "sessionSec" not in gemini
    assert gpt["sessionSec"] == 61.23 and "inputAudioTokens" not in gpt and "cachedInputTokens" not in gpt
    assert r["realtimeSec"] == 61.23


def test_scripted_agent_reports_its_line_cache():
    audio = SimpleNamespace(stats={"linesFromCache": 3, "linesSynthesized": 1, "charactersSynthesized": 40, "linesPlayed": 2})
    r = usage_report(_session(), SimpleNamespace(_audio=audio), [])
    assert r["scriptCache"]["linesFromCache"] == 3 and r["llm"]["inputTokens"] == 0


def test_unreadable_usage_gives_none():
    class Broken:
        @property
        def usage(self):
            raise RuntimeError("closed")

    assert usage_report(Broken(), None, []) is None
