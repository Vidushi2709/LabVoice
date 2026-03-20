import logging

from livekit.agents import AgentSession
from livekit.plugins import deepgram, openai
from livekit.plugins.silero import VAD
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from config import DEEPGRAM_API_KEY, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)


# build session
def build_session() -> AgentSession:
    """Construct and return a fully configured AgentSession."""
    return AgentSession(
        # STT: Deepgram transcribes live audio
        stt=deepgram.STT(
            api_key=DEEPGRAM_API_KEY,
            model="nova-3",
        ),

        # LLM: OpenRouter with conversation history
        llm=openai.LLM(
            model="mistralai/mistral-small-3.2-24b-instruct",
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        ),

        # TTS: Deepgram speaks the response
        tts=deepgram.TTS(
            api_key=DEEPGRAM_API_KEY,
            model="aura-2-thalia-en",
        ),

        vad=VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        min_endpointing_delay=0.2,
    )
