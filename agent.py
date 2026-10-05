import asyncio
import json
import logging
import os
import threading
from collections.abc import Callable

from dotenv import load_dotenv

load_dotenv()  # before importing reporter, which reads BACKEND_URL / AGENT_API_SECRET

from livekit import api, rtc  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.events import CloseEvent, ConversationItemAddedEvent  # noqa: E402
from livekit.plugins import silero  # noqa: E402

from api_tools import build_api_tools  # noqa: E402
from end_call import CallEnder, end_call_tool, hang_up_after_goodbye  # noqa: E402
from call_agent import CallAgent  # noqa: E402
from capacity import worker_options  # noqa: E402
from models import build, is_realtime, opening_line_instructions, realtime_call_config, speech_config  # noqa: E402
from recording import CallRecording, flush_pending, upload  # noqa: E402
from reporter import CallReporter, iso  # noqa: E402
from silence import SilenceWatch  # noqa: E402
from turn_taking import turn_handling, vad_options  # noqa: E402
from usage import usage_report  # noqa: E402

logger = logging.getLogger("voice-agent")


class _ShowProviderErrorDetail(logging.Filter):
    """LiveKit's OpenAI plugin logs the provider's error body only as hidden log metadata
    ("lk.pii.error"), so plain log lines just say "gpt-live returned an error". Put it in the
    message so the reason (bad key, no model access, quota, …) shows up in the logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        detail = getattr(record, "lk.pii.error", None)
        if detail:
            record.msg = f"{record.getMessage()}: {detail}"
            record.args = ()
        return True


logging.getLogger("livekit.plugins.openai").addFilter(_ShowProviderErrorDetail())

AGENT_NAME = os.getenv("AGENT_NAME", "voice-agent")
# Longest a call may run once answered: the agent's own limit (sent per call), never above 10 min.
# MAX_CALL_SECONDS is used when the backend doesn't send one.
MAX_CALL_CAP_SECONDS = 600
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "600"))
# How long a browser call waits for the browser to join its room
WEB_JOIN_TIMEOUT_SECONDS = int(os.getenv("WEB_JOIN_TIMEOUT_SECONDS", "60"))
# Time a finished call gets to report, finish its recording and upload it
JOB_SHUTDOWN_SECONDS = float(os.getenv("JOB_SHUTDOWN_SECONDS", "120"))
DEFAULT_PROMPT = "You are a helpful, friendly voice assistant. Keep your answers short and conversational."
# Opt-in: agents with an enabled script answer matched turns with pre-synthesized lines
SCRIPTED_REPLIES = os.getenv("SCRIPTED_REPLIES") == "1"

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(**vad_options())
    # Recordings an earlier process couldn't upload (backend down, crash) get another try
    threading.Thread(target=lambda: asyncio.run(flush_pending()), name="recording-retry", daemon=True).start()
    if SCRIPTED_REPLIES:
        try:
            from scripted.router import load_model

            load_model()
        except Exception:
            logger.exception("could not preload the script embedding model; it will load on first use")


def call_limit_seconds(config: dict) -> int:
    """The call's time limit: the agent's setting, else MAX_CALL_SECONDS, capped at 10 minutes."""
    try:
        value = int(config.get("maxCallSec") or MAX_CALL_SECONDS)
    except (TypeError, ValueError):
        value = MAX_CALL_SECONDS
    return max(30, min(value, MAX_CALL_CAP_SECONDS))


def wants_script(config: dict) -> bool:
    return SCRIPTED_REPLIES and isinstance(config.get("script"), dict)


def session_options(config: dict) -> dict:
    """AgentSession options for this call: turn-taking tuned for phone calls (see turn_taking.py).
    A realtime model takes its own turns, so it keeps the framework defaults."""
    if is_realtime(config):
        return {}
    # Scripted agents skip preemptive generation: on scripted turns that reply is thrown away,
    # so it would only cost an LLM (and TTS) request per turn.
    stt_provider = speech_config(config)["stt"]["provider"]
    return {"turn_handling": turn_handling(stt_provider, scripted=wants_script(config))}


def build_agent(
    prompt: str,
    greeting: str,
    config: dict,
    session: AgentSession,
    ender: CallEnder | None = None,
    log_transcript: Callable[[dict], None] | None = None,
) -> Agent:
    """The plain LLM agent, or the scripted one when it's switched on for this agent."""
    # Every call can be hung up by the agent itself
    tools = [end_call_tool(ender, config.get("endCallMessage") or "")] if ender else []
    # The agent's external API tools; their calls are logged into the transcript
    tools += build_api_tools(config.get("tools"), log_transcript or (lambda _entry: None))
    script = config.get("script")
    if wants_script(config):
        try:
            from scripted import build_scripted_agent

            agent = build_scripted_agent(
                prompt=prompt, script=script, greeting=greeting, session=session, config=config, tools=tools, ender=ender
            )
            agent.set_filler_words(config.get("fillerWords"))
            agent.log_transcript = log_transcript
            return agent
        except Exception:
            logger.exception("scripted replies failed to start; using the LLM only")
    agent = CallAgent(instructions=prompt, tools=tools)
    # "haan", "hmm"… while the agent talks don't interrupt it
    agent.set_filler_words(config.get("fillerWords"))
    # Notes (ignored fillers, blocked raw output) go to the transcript only, never to the LLM
    agent.log_transcript = log_transcript
    return agent


async def entrypoint(ctx: JobContext):
    # Config sent by the backend via agent dispatch metadata
    try:
        config = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        logger.warning("Invalid job metadata, using defaults: %r", ctx.job.metadata)
        config = {}

    call_id = config.get("callId")
    outbound = config.get("direction") == "outbound"
    # Realtime calls drop the word-for-word lines and pipeline-only features (see models.py)
    config = realtime_call_config(config)
    realtime = is_realtime(config)
    prompt = config.get("prompt") or DEFAULT_PROMPT
    greeting = (config.get("greeting") or "").strip()
    # Language and the STT / LLM / TTS chosen on the agent (see models.py)
    speech = speech_config(config)

    logger.info(
        "Starting call=%s room=%s %s agent=%s (%s) language=%s models=%s",
        call_id,
        ctx.room.name,
        f"outbound to {config.get('phoneNumber')}" if outbound else "web",
        config.get("agentId"),
        config.get("agentName"),
        config.get("language") or "en",
        f"realtime {speech['realtime']['provider']}/{speech['realtime']['model']} voice={speech['realtime']['voiceId']}"
        if realtime
        else f"stt {speech['stt']['provider']}/{speech['stt']['model']}, llm {speech['llm']['provider']}/{speech['llm']['model']}, tts {speech['tts']['provider']}/{speech['tts']['model']}",
    )

    reporter = CallReporter(call_id)
    transcript: list[dict] = []
    state = {"answered": not outbound, "failed": False, "finalized": False}
    # Set once the callee is on the line, if the backend asked for this call to be recorded
    recording: CallRecording | None = None
    # The session and agent, once created (the usage report reads them at the end)
    parts: dict = {}
    # The end_call tool (and scripted "end" scenarios) hang up by shutting the job down
    ender = CallEnder(lambda reason: ctx.shutdown(reason=reason))

    async def finalize(reason: str = ""):
        """Runs once when the job shuts down: final status + transcript to the backend."""
        if state["finalized"]:
            return
        state["finalized"] = True
        try:
            if state["failed"]:
                pass  # already reported
            elif state["answered"]:
                usage = usage_report(parts["session"], parts.get("agent"), transcript) if "session" in parts else None
                await reporter.send("ended", reason=reason or None, transcript=transcript, usage=usage)
            else:
                await reporter.send("ended", reason="Not answered")
            # Hang up the phone leg too (otherwise the callee could be left on silence).
            # When the agent hangs up a browser call, its leaving the room ends the call there.
            if outbound:
                try:
                    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
                except Exception:
                    pass
            if recording and (path := await recording.finish()):
                await upload(path)
                await flush_pending(budget_sec=20)
        finally:
            await reporter.aclose()

    ctx.add_shutdown_callback(finalize)

    await ctx.connect()

    # VAD stays on in realtime mode too: it lets the framework notice the caller barging in
    models = (
        {"llm": build("realtime", speech["realtime"])}
        if realtime
        else {"stt": build("stt", speech["stt"]), "llm": build("llm", speech["llm"]), "tts": build("tts", speech["tts"])}
    )
    session = AgentSession(vad=ctx.proc.userdata["vad"], **models, **session_options(config))

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent):
        item = ev.item
        if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
            return
        text = (item.text_content or "").strip()
        if text:
            entry = {"role": item.role, "text": text, "at": iso(item.created_at)}
            tag = getattr(agent, "transcript_tag", None)  # scripted agents only
            if tag:
                entry.update(tag(item))
            transcript.append(entry)

    @session.on("close")
    def _on_close(ev: CloseEvent):
        # Callee hung up / browser left → end the job, which runs finalize()
        ctx.shutdown(reason=str(getattr(ev.reason, "value", ev.reason)))

    agent = build_agent(prompt, greeting, config, session, ender, log_transcript=transcript.append)
    parts.update(session=session, agent=agent)

    if outbound:
        if not await dial(ctx, session, agent, config, reporter, state):
            ctx.shutdown(reason="dial failed")
            return
    else:
        # Browser calls: the room is created before the browser joins; don't wait for a tab
        # that never arrives (closed page, public-link misuse)
        try:
            await asyncio.wait_for(ctx.wait_for_participant(), WEB_JOIN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.info("call %s: nobody joined within %d s, ending", call_id, WEB_JOIN_TIMEOUT_SECONDS)
            ctx.shutdown(reason="Nobody joined the call")
            return
        await session.start(room=ctx.room, agent=agent)
    ender.mark_live()

    if config.get("record") and call_id:
        try:
            recording = CallRecording(call_id)
            await recording.start(session)
        except Exception:
            logger.exception("could not start recording call %s; the call continues unrecorded", call_id)
            recording = None

    # Asks "are you still there?" when the caller goes quiet, then hangs up
    silence = SilenceWatch(session, ender, config.get("silence"), goodbye=config.get("endCallMessage") or "", realtime=realtime)
    silence.attach()

    # Safety net against calls that never end
    # Time limit, counted from when the callee answered
    limit = call_limit_seconds(config)

    async def max_duration_guard():
        await asyncio.sleep(limit)
        if ender.ending:
            return
        logger.info("call %s reached its %d s limit, ending", call_id, limit)
        silence.stop()
        hang_up_after_goodbye(session, ender, config.get("endCallMessage") or "", f"Max call duration reached ({limit // 60} min {limit % 60} s)".replace(" 0 s", ""))

    guard = asyncio.create_task(max_duration_guard())

    async def stop_guard():
        guard.cancel()
        silence.stop()

    ctx.add_shutdown_callback(stop_guard)

    if greeting and realtime:
        await session.generate_reply(instructions=opening_line_instructions(greeting))
    elif greeting:
        say_line = getattr(agent, "say_line", None)  # scripted agents play the cached greeting
        await (say_line(greeting) if say_line else session.say(greeting))
    else:
        instructions = "Greet the user briefly and ask how you can help, in the conversation language."
        # The opening line must never hang up; a realtime model takes no per-reply tool choice
        await session.generate_reply(instructions=instructions, **({} if realtime else {"tool_choice": "none"}))


async def dial(ctx: JobContext, session: AgentSession, agent: Agent, config: dict, reporter: CallReporter, state: dict) -> bool:
    """Places the outbound call through the SIP trunk. Returns True once the callee answers."""
    phone = config.get("phoneNumber")
    trunk = config.get("sipTrunkId")
    identity = config.get("participantIdentity") or f"sip-{phone}"
    if not phone or not trunk:
        state["failed"] = True
        await reporter.send("failed", reason="Missing phone number or SIP trunk in job metadata")
        return False

    await reporter.send("dialing")

    # Report "ringing" as soon as the SIP participant says so
    def check_ringing(p: rtc.Participant):
        if p.identity == identity and p.attributes.get("sip.callStatus") == "ringing":
            asyncio.create_task(reporter.send("ringing"))

    ctx.room.on("participant_connected", check_ringing)
    ctx.room.on("participant_attributes_changed", lambda _changed, p: check_ringing(p))

    # No speech-to-text while it rings: no caller audio reaches the session and the STT
    # connection (billed) isn't opened until the callee answers
    session.input.set_audio_enabled(False)
    agent.hold_stt_until_answered()

    # Start the session first so it's ready the moment the callee picks up
    session_started = asyncio.create_task(
        session.start(room=ctx.room, agent=agent, room_input_options=RoomInputOptions(participant_identity=identity))
    )

    # Caller ID: the org's number chosen for this call (the trunk's own number when absent)
    from_number = config.get("fromNumber")

    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=trunk,
                sip_call_to=phone,
                **({"sip_number": from_number} if from_number else {}),
                participant_identity=identity,
                participant_name=phone,
                wait_until_answered=True,
            )
        )
    except api.ServerError as e:
        code = getattr(e, "sip_status_code", None)
        logger.warning("call to %s failed: %s", phone, e)
        state["failed"] = True
        await reporter.send(
            "failed",
            sipStatusCode=code,
            sipStatus=getattr(e, "sip_status", None),
            reason=e.message or "Call failed",
        )
        session_started.cancel()
        return False
    except Exception as e:
        logger.exception("call to %s failed", phone)
        state["failed"] = True
        await reporter.send("failed", reason=f"Dial error: {e}"[:500])
        session_started.cancel()
        return False

    await session_started
    session.input.set_audio_enabled(True)
    agent.callee_answered()
    state["answered"] = True
    await reporter.send("answered")
    logger.info("call to %s answered", phone)
    return True


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,  # enables explicit dispatch from the backend
            shutdown_process_timeout=JOB_SHUTDOWN_SECONDS,
            # Calls per worker, warm processes, drain on shutdown, health check port
            **worker_options(),
        )
    )
