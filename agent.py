import asyncio
import json
import logging
import os

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
from livekit.plugins import cartesia, deepgram, openai, silero  # noqa: E402

from reporter import CallReporter, iso  # noqa: E402

logger = logging.getLogger("voice-agent")

AGENT_NAME = os.getenv("AGENT_NAME", "voice-agent")
# Hard stop for any single call (seconds)
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "1800"))
DEFAULT_PROMPT = "You are a helpful, friendly voice assistant. Keep your answers short and conversational."
# Opt-in: agents with an enabled script answer matched turns with pre-synthesized lines
SCRIPTED_REPLIES = os.getenv("SCRIPTED_REPLIES") == "1"

# Languages Cartesia Sonic can speak. Anything else falls back to English.
CARTESIA_LANGUAGES = {
    "en", "fr", "de", "es", "pt", "zh", "ja", "hi", "it",
    "ko", "nl", "pl", "ru", "sv", "tr",
}


def to_cartesia_language(deepgram_language: str) -> str:
    """Map a Deepgram language code (e.g. 'en-US', 'multi') to a Cartesia one (e.g. 'en')."""
    base = deepgram_language.split("-")[0].lower()
    return base if base in CARTESIA_LANGUAGES else "en"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    if SCRIPTED_REPLIES:
        try:
            from scripted.router import load_model

            load_model()
        except Exception:
            logger.exception("could not preload the script embedding model; it will load on first use")


def wants_script(config: dict) -> bool:
    return SCRIPTED_REPLIES and isinstance(config.get("script"), dict)


def session_options(config: dict) -> dict:
    """Extra AgentSession options; none for plain agents."""
    if wants_script(config):
        # Preemptive generation starts an LLM reply before the turn ends; on scripted turns that
        # reply is thrown away, so it would only cost an LLM (and TTS) request per turn.
        return {"turn_handling": {"preemptive_generation": {"enabled": False}}}
    return {}


def build_agent(prompt: str, greeting: str, config: dict, session: AgentSession) -> Agent:
    """The plain LLM agent, or the scripted one when it's switched on for this agent."""
    script = config.get("script")
    if wants_script(config):
        try:
            from scripted import build_scripted_agent

            return build_scripted_agent(prompt=prompt, script=script, greeting=greeting, session=session, config=config)
        except Exception:
            logger.exception("scripted replies failed to start; using the LLM only")
    return Agent(instructions=prompt)


async def entrypoint(ctx: JobContext):
    # Config sent by the backend via agent dispatch metadata
    try:
        config = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        logger.warning("Invalid job metadata, using defaults: %r", ctx.job.metadata)
        config = {}

    call_id = config.get("callId")
    outbound = config.get("direction") == "outbound"
    prompt = config.get("prompt") or DEFAULT_PROMPT
    greeting = (config.get("greeting") or "").strip()
    voice_id = config.get("voiceId")
    language = config.get("language") or "en"

    logger.info(
        "Starting call=%s room=%s %s agent=%s (%s) language=%s",
        call_id,
        ctx.room.name,
        f"outbound to {config.get('phoneNumber')}" if outbound else "web",
        config.get("agentId"),
        config.get("agentName"),
        language,
    )

    reporter = CallReporter(call_id)
    transcript: list[dict] = []
    state = {"answered": not outbound, "failed": False, "finalized": False}

    async def finalize(reason: str = ""):
        """Runs once when the job shuts down: final status + transcript to the backend."""
        if state["finalized"]:
            return
        state["finalized"] = True
        try:
            if state["failed"]:
                pass  # already reported
            elif state["answered"]:
                await reporter.send("ended", reason=reason or None, transcript=transcript)
            else:
                await reporter.send("ended", reason="Not answered")
            # Hang up the phone leg too (otherwise the callee could be left on silence)
            if outbound:
                try:
                    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
                except Exception:
                    pass
        finally:
            await reporter.aclose()

    ctx.add_shutdown_callback(finalize)

    await ctx.connect()

    tts_kwargs = {"model": "sonic-2", "language": to_cartesia_language(language)}
    if voice_id:
        tts_kwargs["voice"] = voice_id

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3", language=language),
        llm=openai.LLM.with_cerebras(model="gpt-oss-120b"),
        tts=cartesia.TTS(**tts_kwargs),
        **session_options(config),
    )

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

    agent = build_agent(prompt, greeting, config, session)

    if outbound:
        if not await dial(ctx, session, agent, config, reporter, state):
            ctx.shutdown(reason="dial failed")
            return
    else:
        await session.start(room=ctx.room, agent=agent)

    # Safety net against calls that never end
    async def max_duration_guard():
        await asyncio.sleep(MAX_CALL_SECONDS)
        logger.info("call %s hit MAX_CALL_SECONDS, ending", call_id)
        ctx.shutdown(reason="Max call duration reached")

    guard = asyncio.create_task(max_duration_guard())

    async def stop_guard():
        guard.cancel()

    ctx.add_shutdown_callback(stop_guard)

    if greeting:
        say_line = getattr(agent, "say_line", None)  # scripted agents play the cached greeting
        await (say_line(greeting) if say_line else session.say(greeting))
    else:
        await session.generate_reply(
            instructions="Greet the user briefly and ask how you can help, in the conversation language."
        )


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
        )
    )
