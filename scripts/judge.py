"""
LLM-as-Judge Module for TradingAgents
=======================================
Post-hoc quality evaluation of analysis results.

Scores on 5 dimensions (1-10):
1. Evidence-Conclusion Alignment
2. Risk Framing
3. Temporal Consistency
4. Contradictory Signal Handling
5. Actionability

Usage:
    from judge import Judge, JudgeResult

    judge = Judge()
    result = judge.evaluate("TSLA", "2026-05-01")
    print(result.overall_score)
    print(result.verdict)      # PASS / CONDITIONAL / REJECT
    print(result.improvements) # Top 3 specific fixes

    # Or evaluate from a loaded dict directly:
    result = judge.evaluate_dict(full_log_data)
"""

import json
import re
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Config — lazy env reading (matches discussion.py pattern)
# ---------------------------------------------------------------------------

def _get_api_key():
    return os.getenv("OPENCODE_GO_API_KEY", "")

def _get_base_url():
    return os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")

def _get_model():
    return os.getenv("JUDGE_MODEL", os.getenv("LANGCHAIN_MODEL_NAME", "glm-5.3-flash"))

def _get_results_dir():
    return Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "/opt/data/workspace/TradingAgents/results"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    name: str
    score: int          # 1-10
    reasoning: str      # 1-2 sentence justification

@dataclass
class JudgeResult:
    ticker: str
    date: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    overall_score: float = 0.0
    verdict: str = "PENDING"     # PASS / CONDITIONAL / REJECT
    improvements: list[str] = field(default_factory=list)
    critique: str = ""           # Full written critique
    evaluated_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def pass_threshold(self) -> float:
        return 7.0

    @property
    def is_pass(self) -> bool:
        return self.overall_score >= self.pass_threshold

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "overall_score": self.overall_score,
            "verdict": self.verdict,
            "dimensions": [{"name": d.name, "score": d.score, "reasoning": d.reasoning} for d in self.dimensions],
            "improvements": self.improvements,
            "critique": self.critique,
            "evaluated_at": self.evaluated_at,
            "duration_s": round(self.duration_s, 1),
            "error": self.error,
        }

    def to_telegram(self) -> str:
        """Plain-text summary for Telegram (no markdown)."""
        lines = [
            f"Judge Evaluation: {self.ticker} ({self.date})",
            f"Overall: {self.overall_score:.1f}/10 — {self.verdict}",
            "",
        ]
        for d in self.dimensions:
            bar = "█" * d.score + "░" * (10 - d.score)
            lines.append(f"  {d.name}: {d.score}/10 {bar}")
            if d.reasoning:
                lines.append(f"    {d.reasoning}")

        if self.improvements:
            lines.append("")
            lines.append("Top improvements:")
            for i, imp in enumerate(self.improvements, 1):
                lines.append(f"  {i}. {imp}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis summarizer — compress ~100K chars into ~8K for the judge
# ---------------------------------------------------------------------------

def summarize_analysis(data: dict) -> str:
    """Extract the key signals from a full_states_log for judge evaluation.
    
    We keep:
    - Final trade decision (full)
    - Investment plan (full)
    - Trader decision (full)
    - Debate outcomes (compressed — just the judge decisions + key points)
    - Analyst summaries (compressed — first 300 chars each)
    
    Total target: ~6000-8000 chars (~2000 tokens)
    """
    parts = []

    # Header
    ticker = data.get("company_of_interest", "?")
    trade_date = data.get("trade_date", "?")
    parts.append(f"TICKER: {ticker}  |  DATE: {trade_date}")

    # Analyst reports — compress to first 300 chars each
    for report_key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
        raw = data.get(report_key, "")
        if raw:
            label = report_key.replace("_report", "").upper()
            compressed = raw[:300].strip()
            if len(raw) > 300:
                compressed += f"... ({len(raw)} chars total)"
            parts.append(f"\n=== {label} ANALYST (summary) ===\n{compressed}")

    # Investment debate — keep judge decision + last responses
    debate = data.get("investment_debate_state", {})
    if debate:
        parts.append("\n=== INVESTMENT DEBATE ===")
        judge = debate.get("judge_decision", "")
        if judge:
            parts.append(f"Judge Decision: {judge[:800]}")
        bull = debate.get("bull_history", "")
        bear = debate.get("bear_history", "")
        if bull:
            # Last 200 chars of bull
            parts.append(f"Bull (last): ...{bull[-200:]}")
        if bear:
            parts.append(f"Bear (last): ...{bear[-200:]}")

    # Trader decision
    trader = data.get("trader_investment_decision", "")
    if trader:
        parts.append(f"\n=== TRADER DECISION ===\n{trader}")

    # Risk debate
    risk = data.get("risk_debate_state", {})
    if risk:
        parts.append("\n=== RISK DEBATE ===")
        risk_judge = risk.get("judge_decision", "")
        if risk_judge:
            parts.append(f"Risk Judge: {risk_judge[:800]}")

    # Investment plan
    plan = data.get("investment_plan", "")
    if plan:
        parts.append(f"\n=== INVESTMENT PLAN ===\n{plan}")

    # Final decision — full
    final = data.get("final_trade_decision", "")
    if final:
        parts.append(f"\n=== FINAL TRADE DECISION ===\n{final}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a senior trading desk manager reviewing an analyst team's work. 
You must evaluate the quality of a multi-agent stock analysis.

Be precise, critical, and constructive. Focus on whether the analysis would actually help a trader make better decisions.

You must respond in EXACTLY this JSON format (no markdown, no code fences):
{
  "dimensions": [
    {"name": "evidence_conclusion", "score": N, "reasoning": "one short sentence"},
    {"name": "risk_framing", "score": N, "reasoning": "one short sentence"},
    {"name": "temporal_consistency", "score": N, "reasoning": "one short sentence"},
    {"name": "contradiction_handling", "score": N, "reasoning": "one short sentence"},
    {"name": "actionability", "score": N, "reasoning": "one short sentence"}
  ],
  "overall_score": N.N,
  "verdict": "PASS|CONDITIONAL|REJECT",
  "improvements": ["one sentence", "one sentence", "one sentence"],
  "critique": "2-3 sentence executive summary"
}

SCORING GUIDE:
- 9-10: Professional-grade, ready to act on
- 7-8: Solid, minor gaps
- 5-6: Notable issues, proceed with caution
- 3-4: Significant flaws, unreliable
- 1-2: Dangerous, do not act on this

DIMENSIONS:
1. evidence_conclusion: Does the final recommendation logically follow from the analyst data? Watch for analysts saying bearish but final decision being bullish without explanation.
2. risk_framing: Are stop-loss levels, position sizing, and downside scenarios realistic and specific? Vague risk statements = low score.
3. temporal_consistency: Does the time horizon match the thesis? (e.g., short-term catalyst with long-term price target = inconsistency)
4. contradiction_handling: When analysts disagree or data conflicts, does the analysis acknowledge and resolve tensions? Ignoring contradictions = low score.
5. actionability: Can a trader execute this today? Specific entry, exit, size, and conditions? "Monitor and wait" without clear triggers = low score.

VERDICT RULES:
- PASS: overall >= 7.5
- CONDITIONAL: overall 5.0-7.4
- REJECT: overall < 5.0

improvements: Exactly 3 specific, actionable improvements the team should make.
critique: 2-4 sentence executive critique summarizing strengths and weaknesses.
"""

JUDGE_USER_PROMPT = """Evaluate this multi-agent trading analysis:

{analysis}"""


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class Judge:
    """LLM-as-Judge evaluator for TradingAgents analysis results."""

    def __init__(self):
        pass

    async def evaluate(self, ticker: str, date: str) -> JudgeResult:
        """Evaluate a completed analysis by ticker + date.

        Reads full_states_log_{date}.json from results dir.
        """
        results_dir = _get_results_dir()
        log_path = results_dir / ticker.upper() / "TradingAgentsStrategy_logs" / f"full_states_log_{date}.json"

        if not log_path.exists():
            return JudgeResult(
                ticker=ticker, date=date,
                error=f"Log file not found: {log_path}"
            )

        with open(log_path) as f:
            data = json.load(f)

        return await self.evaluate_dict(data, ticker, date)

    async def evaluate_dict(self, data: dict, ticker: str = "?", date: str = "?") -> JudgeResult:
        """Evaluate from a loaded full_states_log dict."""
        start = time.time()
        result = JudgeResult(ticker=ticker, date=date)

        # Extract ticker/date from data if not provided
        if ticker == "?":
            ticker = data.get("company_of_interest", "?")
            result.ticker = ticker
        if date == "?":
            date = data.get("trade_date", "?")
            result.date = date

        # Summarize the analysis
        summary = summarize_analysis(data)

        # Call LLM
        try:
            raw_response = await self._call_llm(summary)
        except Exception as e:
            result.error = str(e)
            result.duration_s = time.time() - start
            return result

        # Parse response
        try:
            parsed = self._parse_response(raw_response)
            result.dimensions = [
                DimensionScore(
                    name=d["name"],
                    score=max(1, min(10, int(d["score"]))),
                    reasoning=d.get("reasoning", "")
                )
                for d in parsed.get("dimensions", [])
            ]
            result.overall_score = round(float(parsed.get("overall_score", 0)), 1)
            result.verdict = parsed.get("verdict", "CONDITIONAL")
            result.improvements = parsed.get("improvements", [])
            result.critique = parsed.get("critique", "")
        except Exception as e:
            result.error = f"Parse error: {e}\nRaw: {raw_response[:500]}"
            result.duration_s = time.time() - start
            return result

        result.duration_s = time.time() - start
        return result

    async def _call_llm(self, analysis_summary: str) -> str:
        """Single LLM call to OpenCode Go API with retry on 503."""
        base_url = _get_base_url()
        api_key = _get_api_key()
        model = _get_model()

        if not api_key:
            raise RuntimeError("OPENCODE_GO_API_KEY not set — cannot run judge")

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(analysis=analysis_summary)},
            ],
            "max_tokens": 4096,
            "temperature": 0.3,  # Low temp for consistent scoring
        }

        from _retry import httpx_retry

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await httpx_retry(client, "POST", url, headers=headers, json=payload)
            if resp.status_code == 429:
                raise RuntimeError("OpenCode Go rate limit exhausted")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from LLM response (handles markdown fences, truncation, etc)."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(cleaned[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Try to repair truncated JSON — find what we can
        if brace_start >= 0:
            fragment = cleaned[brace_start:]
            # Try adding closing braces/brackets to complete the JSON
            for suffix in ["}", "]}", "\"]}", "\"]}", "\"}]", "\"}]}", "\"]}}", "\"]},\"critique\":\"\"}"]:
                try:
                    return json.loads(fragment + suffix)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Cannot extract JSON from response ({len(raw)} chars): {raw[:300]}")

    def save_result(self, result: JudgeResult) -> Path:
        """Save judge result alongside the analysis log."""
        results_dir = _get_results_dir()
        out_dir = results_dir / result.ticker.upper() / "TradingAgentsStrategy_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"judge_score_{result.date}.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return out_path

    def load_result(self, ticker: str, date: str) -> Optional[JudgeResult]:
        """Load a previously saved judge result."""
        results_dir = _get_results_dir()
        path = results_dir / ticker.upper() / "TradingAgentsStrategy_logs" / f"judge_score_{date}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        result = JudgeResult(
            ticker=data["ticker"],
            date=data["date"],
            overall_score=data["overall_score"],
            verdict=data["verdict"],
            improvements=data.get("improvements", []),
            critique=data.get("critique", ""),
            evaluated_at=data.get("evaluated_at", 0),
            duration_s=data.get("duration_s", 0),
            error=data.get("error"),
        )
        for d in data.get("dimensions", []):
            result.dimensions.append(DimensionScore(
                name=d["name"],
                score=d["score"],
                reasoning=d.get("reasoning", "")
            ))
        return result


# ---------------------------------------------------------------------------
# Synchronous wrapper for CLI use
# ---------------------------------------------------------------------------

def run_judge(ticker: str, date: str, save: bool = True) -> JudgeResult:
    """Synchronous entry point for CLI / testing."""
    import asyncio

    async def _run():
        judge = Judge()
        result = await judge.evaluate(ticker, date)
        if save and not result.error:
            path = judge.save_result(result)
            print(f"Saved: {path}", flush=True)
        return result

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Load .env for CLI usage
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    if len(sys.argv) < 3:
        print("Usage: python judge.py TICKER DATE")
        print("  python judge.py TSLA 2026-05-01")
        sys.exit(1)

    ticker = sys.argv[1]
    date = sys.argv[2]

    print(f"Evaluating {ticker} ({date})...", flush=True)
    result = run_judge(ticker, date)

    if result.error:
        print(f"\nError: {result.error}")
        sys.exit(1)

    print(f"\n{result.to_telegram()}")
