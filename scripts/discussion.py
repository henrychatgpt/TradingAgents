"""
Discussion Module for TradingAgents Telegram Bot
=================================================
Hybrid Q&A + mini-debate on completed analyses.

- Simple questions  → Q&A mode (1 LLM call, ~5-10s)
- Challenges        → Mini-debate (Bull vs Bear vs Moderator, ~3-5 min)
- Detection         → Heuristic keyword matching (no extra API call)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _get_api_key():
    return os.getenv("OPENCODE_GO_API_KEY", "")

def _get_base_url():
    return os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")

def _get_model():
    return os.getenv("LANGCHAIN_MODEL_NAME", "glm-5.3-flash")

def _get_deepseek_key():
    return os.getenv("DEEPSEEK_API_KEY", "")

def _get_deepseek_base_url():
    return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

DEEPSEEK_MODEL = "deepseek-chat"  # api.deepseek.com does not serve deepseek-v4-flash


# ---------------------------------------------------------------------------
# Discussion State
# ---------------------------------------------------------------------------

@dataclass
class DiscussionState:
    ticker: str
    analysis_context: str  # final trade decision text
    full_log_path: str     # path to full_states_log for deeper lookup
    history: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


# Global per-user discussion state: { user_id: DiscussionState }
discussions: dict[int, DiscussionState] = {}


# ---------------------------------------------------------------------------
# Challenge Detection (heuristic — no extra API call)
# ---------------------------------------------------------------------------

CHALLENGE_PHRASES = [
    "disagree", "wrong", "incorrect", "unrealistic",
    "too conservative", "too aggressive", "too optimistic", "too pessimistic",
    "should be higher", "should be lower", "should be more", "should be less",
    "missing the point", "overlooked", "ignoring", "doesn't account for",
    "not considering", "wrong about", "i'd argue", "challenge that",
    "reconsider", "rethink", "don't agree", "do not agree",
    "not convinced", "not sure about", "i think it should",
    "undervalued", "overvalued", "should be a buy", "should be a sell",
    "too tight", "too loose", "price target is too",
    "stop loss is too", "the stop should",
]


def is_challenge(text: str) -> bool:
    """Detect if user is challenging the analysis."""
    lower = text.lower()
    return any(phrase in lower for phrase in CHALLENGE_PHRASES)

"""OpenCode Go LLM Helper for discussion module."""

import asyncio

import httpx

from _retry import httpx_retry


async def opencode_go_chat(messages: list[dict], max_tokens: int = 4096) -> str:
    """Single LLM call with retry; falls back to DeepSeek direct when the
    primary provider (opencode-go) is exhausted/failing. This is the Q&A /
    mini-debate path used by the bot's discussion mode — without the fallback
    a quota-exhausted primary would retry 5x30s and then fail (bot appears
    unresponsive)."""
    try:
        return await _chat(messages, max_tokens,
                           url=f"{_get_base_url()}/chat/completions",
                           api_key=_get_api_key(), model=_get_model(),
                           max_retries=1)
    except Exception as e:
        print(f"[FALLBACK] opencode-go failed ({type(e).__name__}: {str(e)[:120]}) — trying DeepSeek", flush=True)
        return await _chat(messages, max_tokens,
                           url=f"{_get_deepseek_base_url()}/chat/completions",
                           api_key=_get_deepseek_key(), model=DEEPSEEK_MODEL,
                           max_retries=2)


async def _chat(messages: list[dict], max_tokens: int, url: str, api_key: str,
                model: str, max_retries: int = 2) -> str:
    """OpenAI-compatible chat completion with bounded retry."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await httpx_retry(client, "POST", url, headers=headers, json=payload,
                                 max_retries=max_retries)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Q&A Mode
# ---------------------------------------------------------------------------

async def answer_question(state: DiscussionState, question: str) -> str:
    """Answer a question using the analysis as context. Single LLM call."""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a team of expert trading analysts who completed "
                f"a comprehensive analysis of {state.ticker}.\n\n"
                f"FULL ANALYSIS:\n{state.analysis_context}\n\n"
                "Answer the user's question based on this analysis. "
                "Reference exact data points and numbers. "
                "Keep response under 500 words. Plain text only."
            )
        }
    ]
    # Include conversation history (last 3 exchanges = 6 messages)
    messages.extend(state.history[-6:])
    messages.append({"role": "user", "content": question})

    return await opencode_go_chat(messages, max_tokens=2000)


# ---------------------------------------------------------------------------
# Mini-Debate Mode (3 agents: Bull, Bear, Moderator)
# ---------------------------------------------------------------------------

async def run_mini_debate(state: DiscussionState, challenge: str) -> str:
    """Run a 3-agent mini-debate on the user's challenge."""

    ctx = (
        f"Original analysis of {state.ticker}:\n{state.analysis_context}\n\n"
        f"User's challenge: {challenge}"
    )

    # Agent 1: Bull Advocate — defend the original analysis
    bull = await opencode_go_chat([
        {
            "role": "system",
            "content": (
                f"You are the Bull Advocate on a trading analysis team.\n{ctx}\n\n"
                "Argue WHY the original analysis position is correct and should be maintained. "
                "Use specific data points from the analysis. Be persuasive but fair. "
                "300 words max. Plain text only."
            )
        },
        {"role": "user", "content": f"Defend the analysis against: {challenge}"}
    ], max_tokens=1500)

    # Agent 2: Bear Advocate — argue the challenge has merit
    bear = await opencode_go_chat([
        {
            "role": "system",
            "content": (
                f"You are the Bear Advocate on a trading analysis team.\n{ctx}\n\n"
                "Argue WHY the original analysis may be WRONG based on the user's challenge. "
                "Use specific data points. Be persuasive but fair. "
                "300 words max. Plain text only."
            )
        },
        {"role": "user", "content": f"Challenge the analysis: {challenge}"}
    ], max_tokens=1500)

    # Agent 3: Moderator — synthesize and give final verdict
    synthesis = await opencode_go_chat([
        {
            "role": "system",
            "content": (
                f"You are the Moderator of a trading debate.\n{ctx}\n\n"
                f"Bull Advocate argues:\n{bull}\n\n"
                f"Bear Advocate argues:\n{bear}\n\n"
                "Synthesize both views and the user's challenge. "
                "Should the original analysis be revised? If so, provide specific updated recommendations "
                "(new price targets, stop-loss levels, position sizing). Be decisive and clear. "
                "400 words max. Plain text only."
            )
        },
        {"role": "user", "content": challenge}
    ], max_tokens=2000)

    return (
        f"🐂 Bull Advocate:\n{bull}\n\n"
        f"🐻 Bear Advocate:\n{bear}\n\n"
        f"⚖️ Synthesis:\n{synthesis}"
    )
