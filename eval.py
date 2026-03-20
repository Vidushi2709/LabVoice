"""
eval.py — All agent quality metrics in one place.

Tracks:
  [Existing]
    • EOU latency        – end-of-utterance delay per turn
    • Token usage        – prompt / completion / total tokens via UsageCollector

  [New]
    • Barge-in rate      – successful_interruptions / total_interruptions
    • Task completion    – LLM-as-judge score (0–1) per conversation turn
    • Repair rate        – turns where user had to repeat themselves / total turns
"""

import asyncio
import difflib
import logging
from dataclasses import dataclass, field
from typing import Optional, cast

import httpx
from livekit.agents import AgentSession, AgentStateChangedEvent, MetricsCollectedEvent, JobContext, metrics

from config import OPENROUTER_API_KEY

logger = logging.getLogger(__name__)


#  Repair heuristics 

# Common voice repair markers — user correcting or repeating themselves
_REPAIR_PHRASES = {
    # Explicit corrections
    "i said", "i meant", "i mean", "what i said", "i just said",
    "i already said", "i told you", "i already told",
    # Negation / redirect
    "no,", "no i", "no no", "not that", "that's not", "that is not",
    "that's not what", "not what i",
    # Wait / rephrase signals (common in voice)
    "wait", "actually", "no wait", "hold on",
    "sorry", "let me rephrase", "let me say",
    # Explicit repeat requests
    "repeat", "again", "didn't you hear", "are you listening",
    "can you hear",
}

# Lower threshold vs text chat: STT often paraphrases slightly,
# so 0.60 catches near-repeats without too many false positives.
_SIMILARITY_THRESHOLD = 0.60


def _is_repair(prev: str, curr: str) -> bool:
    """Heuristic: did the user repeat / correct themselves?"""
    if not prev:
        return False
    curr_lower = curr.lower()
    # 1. Explicit repair phrase
    if any(phrase in curr_lower for phrase in _REPAIR_PHRASES):
        return True
    # 2. Lexical similarity — user said roughly the same thing again
    ratio = difflib.SequenceMatcher(None, prev.lower(), curr_lower).ratio()
    return ratio > _SIMILARITY_THRESHOLD


# LLM-as-judge 

_JUDGE_PROMPT = """\
You are evaluating a voice assistant response. Answer ONLY with 0 or 1.

1 = the assistant directly answered the user's question or completed their request
0 = the assistant failed, deflected, asked a clarifying question, or gave an irrelevant response

User said: {user_text}
Assistant said: {agent_text}

Answer (0 or 1 only):\
"""


async def _llm_judge(user_msg: str, agent_response: str) -> float:
    """Call OpenRouter to score how well the agent completed the user's task."""
    if not user_msg or not agent_response:
        return -1.0

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": [
                        {
                            "role": "user",
                            "content": _JUDGE_PROMPT.format(user_text=user_msg, agent_text=agent_response),
                        },
                    ],
                    "max_tokens": 5,
                    "temperature": 0,
                },
            )
        if r.status_code != 200:
            logger.warning("LLM judge HTTP %s: %s", r.status_code, r.text[:120])
            return -1.0
        try:
            data = cast(dict, r.json())
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            text = str(content).strip()
        except Exception:
            text = ""
            
        text_clean = text.strip().replace(".", "")
        if text_clean not in ("0", "1"):
            logger.warning("Judge returned unexpected output: '%s'", text)
            return -1.0
        return float(text_clean)
    except Exception as exc:
        logger.warning("LLM judge error: %s", exc)
        return -1.0


# EvalTracker 

class EvalTracker:
    """Central tracker for every agent quality metric."""

    def __init__(self) -> None:
        #  Barge-in 
        self.total_interruptions: int = 0
        self.successful_interruptions: int = 0
        self._interrupted: bool = False          # True while we're waiting to see if the barge-in was acted on

        #  Repair rate 
        self.total_turns: int = 0
        self.repair_turns: int = 0
        self._last_user_msg: str = ""

        #  Task completion (LLM judge) 
        self._task_scores: list[float] = field(default_factory=list) if False else []

        #  Existing metrics 
        self._usage_collector = metrics.UsageCollector()
        self._eou_latencies: list[float] = []
        self._last_eou: Optional[metrics.EOUMetrics] = None

    #  Attach to a live session 

    def attach(self, session: AgentSession, ctx: JobContext) -> None:
        """Wire all metric collection hooks to *session*."""

        @session.on("metrics_collected")
        def _on_metrics(ev: MetricsCollectedEvent):
            m = ev.metrics
            #  Existing: EOU latency
            if m.type == "eou_metrics":
                self._last_eou = m
                # field name differs slightly across SDK versions
                delay = getattr(m, "end_of_utterance_delay", None) or getattr(m, "eou_delay", None)
                if delay is not None:
                    self._eou_latencies.append(delay)
                # Log EOU alongside the speech that triggered it
                if session.current_speech and getattr(m, "speech_id", None) == session.current_speech.id:
                    logger.info("EOU metrics: %s", m.model_dump())
            #  Existing: forward all metrics to SDK logger + usage collector 
            metrics.log_metrics(m)
            self._usage_collector.collect(m)

        @session.on("agent_state_changed")
        def _on_state_changed(ev: AgentStateChangedEvent):
            old, new = str(ev.old_state), str(ev.new_state)
            logger.info("Agent state: %s → %s", old, new)
            self._track_barge_in(old, new)

        ctx.add_shutdown_callback(self._log_final_summary)

    #  Barge-in tracking 

    def _track_barge_in(self, old: str, new: str) -> None:
        if old == "speaking" and new == "listening":
            # User started talking while agent was speaking → potential barge-in
            self.total_interruptions += 1
            self._interrupted = True
            logger.info("Barge-in detected  [total=%d]", self.total_interruptions)

        elif self._interrupted and new == "thinking":
            # Agent moved to thinking → it captured the interruption → SUCCESS
            self.successful_interruptions += 1
            self._interrupted = False
            logger.info(
                "Barge-in SUCCESS  [rate=%.0f%%  %d/%d]",
                self.barge_in_success_rate * 100,
                self.successful_interruptions,
                self.total_interruptions,
            )

        elif self._interrupted and new == "speaking":
            # Agent resumed speaking without acting on the interruption → FAILED
            self._interrupted = False
            logger.info(
                "Barge-in FAILED   [rate=%.0f%%  %d/%d]",
                self.barge_in_success_rate * 100,
                self.successful_interruptions,
                self.total_interruptions,
            )

    #  Per-turn evaluation (called from Assistant.on_llm_request) 

    async def on_turn(self, user_msg: str, agent_response: str) -> None:
        """
        Call once per user→agent exchange (fire-and-forget via asyncio.create_task).
        Runs repair detection and the LLM-as-judge concurrently.
        """
        #  Repair rate 
        self.total_turns += 1
        if _is_repair(self._last_user_msg, user_msg):
            self.repair_turns += 1
            logger.info(
                "Repair detected   [turn=%d  rate=%.0f%%]",
                self.total_turns,
                self.repair_rate * 100,
            )
        self._last_user_msg = user_msg

        #  LLM-as-judge (async, doesn't block the main pipeline) 
        score = await _llm_judge(user_msg, agent_response)
        if score >= 0:
            self._task_scores.append(score)
            logger.info(
                "Task completion   [score=%.2f  avg=%.2f  n=%d]",
                score,
                self.avg_task_completion,
                len(self._task_scores),
            )

    #  Computed properties 

    @property
    def barge_in_success_rate(self) -> float:
        return self.successful_interruptions / self.total_interruptions if self.total_interruptions else 0.0

    @property
    def repair_rate(self) -> float:
        return self.repair_turns / self.total_turns if self.total_turns else 0.0

    @property
    def avg_task_completion(self) -> float:
        valid = [s for s in self._task_scores if s >= 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_eou_latency(self) -> Optional[float]:
        return sum(self._eou_latencies) / len(self._eou_latencies) if self._eou_latencies else None

    #  Final summary 

    async def _log_final_summary(self) -> None:
        usage = self._usage_collector.get_summary()
        eou   = f"{self.avg_eou_latency:.3f}s" if self.avg_eou_latency is not None else "N/A"

        logger.info(
            "\n"
            "════════════════════ EVAL SUMMARY ════════════════════\n"
            "  Barge-in success rate : %.0f%%  (%d successful / %d total)\n"
            "  Avg task completion   : %.2f / 1.00  (%d turns scored)\n"
            "  Repair rate           : %.0f%%  (%d repairs / %d turns)\n"
            "  Avg EOU latency       : %s\n"
            "  Token usage           : %s\n"
            "══════════════════════════════════════════════════════",
            self.barge_in_success_rate * 100,
            self.successful_interruptions,
            self.total_interruptions,
            self.avg_task_completion,
            len(self._task_scores),
            self.repair_rate * 100,
            self.repair_turns,
            self.total_turns,
            eou,
            usage,
        )
