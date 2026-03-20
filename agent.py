import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import os

from dotenv import load_dotenv
load_dotenv()
from livekit.agents import(
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    AgentStateChangedEvent,
    MetricsCollectedEvent,
    metrics,
    function_tool,
)
from livekit.agents.llm import ChatContext
from livekit.plugins import noise_cancellation, deepgram, openai
from livekit.plugins.silero import VAD
from livekit.plugins.turn_detector.multilingual import MultilingualModel

MAX_MESSAGES = 10 
DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

# tool calls

# basic tavily search 
@function_tool
@function_tool
async def search_web(query: str) -> str:
    """Search the web for ML papers, docs, benchmarks, or any current information"""
    logger.info("TOOL CALLED: search_web | query: %s", query)
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={                              # ← Tavily uses POST with JSON body
                "api_key": os.getenv("TAVILY_API_KEY"),
                "query": query,
                "max_results": 3,
            }
        )
        
        # Debug: see exactly what Tavily returned
        logger.info("Tavily status: %s | body: %s", r.status_code, r.text[:300])
        
        if r.status_code != 200:
            return f"Search failed: {r.status_code} - {r.text[:100]}"
        
        data = r.json()
        results = data.get("results", [])
        
        if not results:
            return f"No results found for: {query}"
        
        # Return top 2 results
        output = []
        for res in results[:2]:
            title = res.get("title", "No title")
            content = res.get("content", "No content")[:200]
            url = res.get("url", "")
            output.append(f"{title}: {content} ({url})")
        
        result = "\n\n".join(output)
        logger.info(" TOOL RESULT: search_web | %s", result[:100])
        return result

# calculator 
@function_tool
async def calculate(expression: str) -> str:
    """Evaluate a math expression — latency, cost, parameter counts etc"""
    logger.info(" TOOL CALLED: calculate | expression: %s", expression)
    try:
        return str(eval(expression))  # safe for simple math
    except Exception as e:
        logger.error(" TOOL ERROR: calculate | error: %s", e)
        return f"Error: {e}"

# arxiv search for papers and abstracts
@function_tool
async def search_arxiv(query: str) -> str:
    """Search arxiv for ML/AI research papers"""
    logger.info(" TOOL CALLED: search_arxiv | query: %s", query)
    import httpx
    import xml.etree.ElementTree as ET
    
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 3}
        )
    
    # Parse XML properly instead of dumping raw XML to LLM
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    
    if not entries:
        return "No papers found."
    
    results = []
    for entry in entries[:3]:
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()[:150]
        link = entry.find("atom:id", ns).text.strip()
        results.append(f"- {title}: {summary}... ({link})")
    
    result = "\n".join(results)
    logger.info("TOOL RESULT: search_arxiv | %s", result[:100])
    return result

# hf model lookup for stats and benchmarks
@function_tool
async def lookup_model(model_name: str) -> str:
    """Look up a HuggingFace model's stats"""
    import httpx
    hf_token = os.getenv("HF_TOKEN") 
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://huggingface.co/api/models/{model_name}",
            headers=headers
        )
        if r.status_code != 200:
            return f"Model not found: {model_name}"
        data = r.json()
        return f"Downloads: {data.get('downloads')}, Likes: {data.get('likes')}, Tags: {data.get('tags')}"
    
# a basic assistant class
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a sharp, witty AI lab partner helping a developer think through ML and systems ideas in real-time. "
                "Your tone is slightly sarcastic, confident, and playful—but never annoying or cringey. "
                
                "Speak like a smart Gen Z engineer: concise, clever, and opinionated. "
                "Keep every response under 2–3 sentences max. No rambling, no lectures."
                
                "Your job is to:"
                "- Challenge bad ideas directly"
                "- Suggest better approaches quickly"
                "- Point out tradeoffs (latency, scale, cost)"
                "- Think like someone building real systems, not theory"
                "If the user is vague, ask a sharp clarifying question instead of guessing. "
                "If something is wrong, say it plainly and explain briefly."
                "Avoid filler, avoid politeness fluff, avoid generic advice."
            ),
            tools = [search_web, calculate, search_arxiv, lookup_model],
        )
    
    # trim messages 
    async def _trim_history(self, agent,chat_ctx: ChatContext, *args, **kwargs):
        messages = chat_ctx.messages
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [
            m for m in chat_ctx.messages
            if m.role != "system" and m.role != "tool"  
        ]
        
        if len(non_system) > MAX_MESSAGES:
            # keep all system messages + the last MAX_MESSAGES non-system messages
            trimmed = system_msgs + non_system[-MAX_MESSAGES:]
            chat_ctx.messages = trimmed
            logger.info(f"Trimmed chat history to last {MAX_MESSAGES} messages (plus system messages)")
        return await super().on_llm_request(chat_ctx, *args, **kwargs)
            
# entrypoint for the agent session
async def entrypoint(ctx: JobContext):
        
    await ctx.connect() # ctx joins an existing room

    # processing start
    session = AgentSession(
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
        turn_detection=MultilingualModel(), # session now emits turn events when the semantic model decides a speaker is finished.
        preemptive_generation=True, # allow the LM to start generating a response before the user has completely finished speaking
        min_endpointing_delay=0.2,
        )
        
        
    usage_collector = metrics.UsageCollector() # collect usage metrics for the agent session
    last_eou_metrics: metrics.EOUMetrics = None # eou = end of utterance
        
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)


    ctx.add_shutdown_callback(log_usage)
    
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        logger.info(f"Agent state changed: {ev.old_state} -> {ev.new_state}")
        
        if (
            str(ev.new_state) == "speaking"
            and last_eou_metrics
            and session.current_speech
            and last_eou_metrics.speech_id == session.current_speech.id
        ):
            # Pydantic model — this will show all fields
            logger.info("EOU metrics: %s", last_eou_metrics.model_dump())
            
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint
        ))