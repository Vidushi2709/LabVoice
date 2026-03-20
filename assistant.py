import asyncio
import logging
from typing import TYPE_CHECKING

from livekit.agents import Agent
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.agent import ModelSettings

from config import MAX_MESSAGES
from tools import search_web, calculate, search_arxiv, lookup_model

if TYPE_CHECKING:
    from eval import EvalTracker          # avoid circular import at runtime

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are a sharp, witty AI lab partner helping a developer think through ML and systems ideas in real-time. "
    "Your tone is slightly sarcastic, confident, and playful—but never annoying or cringey. "

    "Speak like a smart Gen Z engineer: concise, clever, and opinionated. "
    "Keep every response under 2–3 sentences max. No rambling, no lectures."

    "Your job is to:"
    "- Challenge bad ideas directly"
    "- Suggest better approaches quickly"
    "- Never ask more than one clarifying question per turn. " 
    "- If you have enough context to help, just help."   
    "- Point out tradeoffs (latency, scale, cost)"
    "- Think like someone building real systems, not theory"
    "- If the user is vague, ask a sharp clarifying question instead of guessing. "
    "- If something is wrong, say it plainly and explain briefly."
    "- Avoid filler, avoid politeness fluff, avoid generic advice."
)


# a basic assistant class
class Assistant(Agent):
    def __init__(self, eval_tracker: "EvalTracker | None" = None) -> None:
        super().__init__(
            instructions=INSTRUCTIONS,
            tools=[search_web, calculate, search_arxiv, lookup_model],
        )
        self._eval_tracker = eval_tracker

    #  History trim + logging 

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        """
        Override llm_node to trim history before sending to the LLM.

        CRITICAL: we work on a COPY of chat_ctx.
        The incoming chat_ctx IS the agent's live _chat_ctx object (confirmed in
        agent_activity.py:961). Calling truncate() on it in-place permanently destroys
        history every turn, shifting the cache prefix and dropping cache hit rates.
        """
        #  1. Work on a copy — never mutate the agent's persistent history 
        trimmed = chat_ctx.copy()

        non_system_before = [
            m for m in trimmed.messages()
            if m.role not in ("system", "developer")
        ]
        logger.info("History BEFORE trim: %d msgs", len(non_system_before))

        trimmed.truncate(max_items=MAX_MESSAGES)

        non_system_after = [
            m for m in trimmed.messages()
            if m.role not in ("system", "developer")
        ]
        logger.info("History AFTER  trim: %d msgs  (limit=%d)", len(non_system_after), MAX_MESSAGES)

        #  2. Pin system message to index 0   
        # OpenRouter/OpenAI cache the prefix. If the system message isn't strictly
        # first — or shifts position due to AgentConfigUpdate items being inserted
        # at session start — every turn has a different prefix → cache miss.
        items = trimmed.items
        sys_idx = next(
            (
                i for i, item in enumerate(items)
                if item.type == "message" and item.role in ("system", "developer")
            ),
            None,
        )

        if sys_idx is None:
            logger.warning("No system/developer message found in chat_ctx!")
        elif sys_idx != 0:
            sys_msg = items.pop(sys_idx)
            items.insert(0, sys_msg)
            logger.warning(
                "System message was at index %d — moved to front to stabilise cache prefix",
                sys_idx,
            )
        # else: already at 0, cache prefix is stable ✓

        #  3. Stream to default LLM handler 
        async for chunk in Agent.default.llm_node(self, trimmed, tools, model_settings):
            yield chunk

    #  Per-turn eval 

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """
        Called when user finishes speaking, right before the LLM call.
        Used exclusively for eval (repair detection + LLM-as-judge).
        Trimming is now done in llm_node where it's guaranteed to take effect.
        """
        if not self._eval_tracker:
            return

        user_msg = new_message.text_content or ""

        # turn_ctx = history before this user message → last assistant msg = previous response
        past_assistant = [m for m in turn_ctx.messages() if m.role == "assistant"]
        agent_response = past_assistant[-1].text_content or "" if past_assistant else ""

        # Fire-and-forget: never delays TTS
        asyncio.create_task(
            self._eval_tracker.on_turn(user_msg, agent_response)
        )
