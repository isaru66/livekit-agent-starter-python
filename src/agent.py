import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    function_tool,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import azure

import asyncio
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tavily import TavilyClient

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."

    @function_tool
    async def search_web(self, query: str) -> str:
        """
        Search the web for information based on the given query.

        Always use this function whenever the user requests a web search

        Args:
            query: The search query to look up on the web.
        """

        # simulate a long web search to demonstrate the background "thinking" audio
        logger.info("TAVILY SearchAgent thinking...")
        
        

        client = TavilyClient(os.environ["TAVILY_API_KEY"])
        response = client.search(
            query=query,
            search_depth="advanced"
        )
        print(response)
        logger.info("TAVILY SearchAgent finished. simulating delay...")
        delay = os.environ.get("TAVILY_SEARCH_DELAY")
        if delay:
            await asyncio.sleep(int(delay))
        return response
        # return "The request failed, give the users some information based on your knowledge"


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # stt=inference.STT(model="cartesia/ink-whisper", language="th"),
        # stt=inference.STT(model="deepgram/nova-2", language="th"),
        stt=azure.stt.STT(
            speech_key=os.environ["AZURE_SPEECH_KEY"],
            speech_region=os.environ["AZURE_SPEECH_REGION"],
            language="th-TH",
        ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/

        # Voice:
        # male: Blake cartesia/sonic-3:a167e0f3-df7e-4d52-a9c3-f949145efdab
        # female: Jacqueline cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc        
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="a167e0f3-df7e-4d52-a9c3-f949145efdab"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        # ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        
        # play ringtone sound when the agent is thinking
        thinking_sound=[
            # AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            # AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
            # AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=0.8),
            AudioConfig("sound/ringtone.mp3", volume=0.7),
        ],
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    # Initiating speech
    await session.say(
        "สวัดดีครับ ต้องการถามอะไรเกี่ยวกับ Contoso Bank ไหมครับ",
        allow_interruptions=True,
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
