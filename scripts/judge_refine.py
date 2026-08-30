"""
Judge Refine Module — Phase 3
================================
Self-critique + refinement loop for TradingAgents analyses.

Takes the judge's specific improvements and feeds them back into a single
LLM call that revises the final trade decision. Costs 1 API call instead of
re-running the full 13-agent pipeline (~25 min).

Workflow:
  1. Load analysis (full_states_log) + judge score
  2. Build refinement prompt: original decision + judge improvements
  3. Single LLM call produces revised decision
  4. Save refined decision alongside original
  5. Re-run judge on refined version to measure improvement

Usage:
    from judge_refine import Refiner
    refiner = Refiner()
    result = refiner.refine("NVDA", "2026-05-01")
    print(result.delta_score)  # How much the score improved

    # CLI:
    python judge_refine.py NVDA 2026-05-01
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Config — lazy env reading
# ---------------------------------------------------------------------------

def _get_api_key():
    return os.getenv("OPENCODE_GO_API_KEY", "")

def _get_base_url():
    return os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")

def _get_model():
    return os.getenv("REFINE_MODEL", os.getenv("LANGCHAIN_MODEL_NAME", "glm-5.3-flash"))

def _get_results_dir():
    return Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "/opt/data/workspace/TradingAgents/results"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RefineResult:
    ticker: str
    date: str
    original_score: float = 0.0
    refined_score: float = 0.0
    delta_score: float = 0.0
    original_verdict: str = ""
    refined_verdict: str = ""
    refined_decision: str = ""     # The revised final trade decision
    improvements_applied: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    judge_duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def improved(self) -> bool:
        return self.delta_score > 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "original_score": self.original_score,
            "refined_score": self.refined_score,
            "delta_score": self.delta_score,
            "original_verdict": self.original_verdict,
            "refined_verdict": self.refined_verdict,
            "refined_decision": self.refined_decision[:2000],  # Cap for storage
            "improvements_applied": self.improvements_applied,
            "duration_s": round(self.duration_s, 1),
            "judge_duration_s": round(self.judge_duration_s, 1),
            "error": self.error,
        }

    def to_telegram(self) -> str:
        """Plain-text summary for Telegram."""
        lines = [
            f"Refinement Report: {self.ticker} ({self.date})",
            "",
            f"Original: {self.original_score:.1f}/10 ({self.original_verdict})",
            f"Refined:  {self.refined_score:.1f}/10 ({self.refined_verdict})",
            f"Delta:    {self.delta_score:+.1f}",
            "",
        ]

        if self.error:
            lines.append(f"Error: {self.error}")
            return "\n".join(lines)

        arrow = "+" if self.delta_score > 0 else ""
        if self.delta_score > 0.5:
            lines.append(f"Refinement successful ({arrow}{self.delta_score:.1f} improvement)")
        elif self.delta_score > 0:
            lines.append(f"Marginal improvement ({arrow}{self.delta_score:.1f})")
        elif self.delta_score == 0:
            lines.append("No score change — original was already strong")
        else:
            lines.append(f"Score decreased ({arrow}{self.delta_score:.1f}) — refinement may have introduced issues")

        if self.improvements_applied:
            lines.append("")
            lines.append("Improvements applied:")
            for i, imp in enumerate(self.improvements_applied, 1):
                lines.append(f"  {i}. {imp}")

        lines.append("")
        lines.append(f"Refine time: {self.duration_s:.0f}s | Judge time: {self.judge_duration_s:.0f}s")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Refinement prompt
# ---------------------------------------------------------------------------

REFINE_SYSTEM_PROMPT = """You are a senior trading desk manager revising a team's analysis based on quality review feedback.

You will receive:
1. The ORIGINAL final trade decision (from the multi-agent analysis)
2. The JUDGE'S CRITIQUE (overall assessment)
3. THREE SPECIFIC IMPROVEMENTS requested by the judge

Your task: Produce a REVISED final trade decision that:
- Addresses ALL three improvements specifically
- Preserves the original recommendation direction (don't flip from Buy to Sell)
- Maintains the same structure and format as the original
- Resolves any contradictions, conflicting numbers, or ambiguities identified
- Keeps the same level of detail and specificity

Output the revised decision in the SAME format as the original. Start with **Rating**: and follow the same structure.
Do NOT add a preamble or explanation of what you changed — just output the revised decision directly.
"""

REFINE_USER_PROMPT = """REVISE THIS ANALYSIS:

=== ORIGINAL FINAL TRADE DECISION ===
{original_decision}

=== JUDGE'S CRITIQUE ===
{critique}

=== THREE REQUIRED IMPROVEMENTS ===
1. {improvement_1}
2. {improvement_2}
3. {improvement_3}

=== OVERALL SCORE ===
{score}/10 ({verdict})

Produce the revised final trade decision now:"""


# ---------------------------------------------------------------------------
# Refiner class
# ---------------------------------------------------------------------------

class Refiner:
    """Self-critique refinement: revise analysis using judge feedback."""

    def __init__(self):
        pass

    @staticmethod
    def _find_latest_with_judge(logs_dir: Path) -> Optional[str]:
        """Find the latest date that has both a full_states_log and judge_score.
        
        Returns the date string (e.g. '2026-05-01') or None.
        """
        if not logs_dir.exists():
            return None
        logs = sorted(logs_dir.glob("full_states_log_*.json"), reverse=True)
        for log in logs:
            # Extract date from filename: full_states_log_2026-05-01.json
            name = log.stem  # full_states_log_2026-05-01
            date = name.replace("full_states_log_", "")
            # Skip refined copies
            if "_refined" in date:
                continue
            # Check judge score exists
            judge_path = logs_dir / f"judge_score_{date}.json"
            if judge_path.exists():
                return date
        return None

    async def refine(self, ticker: str, date: str, force: bool = False) -> RefineResult:
        """Refine an analysis by applying judge feedback.
        
        Only refines if score < REFINE_THRESHOLD (default 7.5) unless force=True.
        Rationale: refinement improves weak analyses but can degrade strong ones.
        
        Steps:
        1. Load analysis + judge score
        2. Call LLM to revise decision
        3. Re-judge the refined version
        4. Compare scores
        5. Keep the BETTER version
        """
        start = time.time()
        results_dir = _get_results_dir()
        result = RefineResult(ticker=ticker, date=date)

        # Load analysis — auto-find latest if specified date not found
        logs_dir = results_dir / ticker.upper() / "TradingAgentsStrategy_logs"
        log_path = logs_dir / f"full_states_log_{date}.json"
        if not log_path.exists():
            # Find latest log that also has a matching judge score
            best = self._find_latest_with_judge(logs_dir)
            if best:
                date = best  # use the found date instead
                result.date = date
                log_path = logs_dir / f"full_states_log_{date}.json"
            else:
                result.error = f"No analysis with judge score found for {ticker}. Run /analyze then /judge first."
                return result

        with open(log_path) as f:
            log_data = json.load(f)

        original_decision = log_data.get("final_trade_decision", "")
        if not original_decision:
            result.error = "No final_trade_decision in log"
            return result

        # Load judge score
        judge_path = results_dir / ticker.upper() / "TradingAgentsStrategy_logs" / f"judge_score_{date}.json"
        if not judge_path.exists():
            result.error = f"Judge score not found. Run /judge {ticker} first."
            return result

        with open(judge_path) as f:
            judge_data = json.load(f)

        result.original_score = judge_data.get("overall_score", 0)
        result.original_verdict = judge_data.get("verdict", "?")
        critique = judge_data.get("critique", "")
        improvements = judge_data.get("improvements", [])

        if len(improvements) < 3:
            result.error = "Judge score has fewer than 3 improvements — cannot refine"
            return result

        result.improvements_applied = improvements

        # Threshold guard: only refine weak analyses unless forced
        REFINE_THRESHOLD = 7.5
        if result.original_score >= REFINE_THRESHOLD and not force:
            result.refined_score = result.original_score
            result.refined_verdict = result.original_verdict
            result.delta_score = 0.0
            result.refined_decision = original_decision
            result.duration_s = time.time() - start
            result.error = f"Score {result.original_score:.1f} >= threshold {REFINE_THRESHOLD}. Use force=True to override."
            return result

        # Step 1: Call LLM to revise
        try:
            refined_decision = await self._call_llm(
                original_decision, critique, improvements,
                result.original_score, result.original_verdict
            )
        except Exception as e:
            result.error = f"Refinement LLM call failed: {e}"
            result.duration_s = time.time() - start
            return result

        result.refined_decision = refined_decision
        refine_time = time.time() - start

        # Step 2: Save refined decision to the log (in a copy, not overwriting original)
        # We create a "refined" version of the log
        refined_log = dict(log_data)
        refined_log["final_trade_decision"] = refined_decision
        refined_log["investment_plan"] = refined_decision  # Also update plan
        refined_log["_refinement_meta"] = {
            "refined_at": time.time(),
            "original_score": result.original_score,
            "improvements_applied": improvements,
        }

        refined_log_path = results_dir / ticker.upper() / "TradingAgentsStrategy_logs" / f"full_states_log_{date}_refined.json"
        with open(refined_log_path, "w") as f:
            json.dump(refined_log, f, indent=2)

        # Step 3: Re-judge the refined version
        from judge import Judge
        judge = Judge()
        judge_start = time.time()
        try:
            new_judge_result = await judge.evaluate_dict(refined_log, ticker, date)
            result.refined_score = new_judge_result.overall_score
            result.refined_verdict = new_judge_result.verdict
            result.delta_score = round(result.refined_score - result.original_score, 1)
        except Exception as e:
            result.error = f"Re-judge failed: {e}"
            result.duration_s = time.time() - start
            return result

        result.judge_duration_s = time.time() - judge_start
        result.duration_s = time.time() - start

        # Step 4: Save refined judge score (NEVER overwrite original)
        if new_judge_result and not new_judge_result.error:
            # Save refined log
            refined_judge_path = results_dir / ticker.upper() / "TradingAgentsStrategy_logs" / f"judge_score_{date}_refined.json"
            with open(refined_judge_path, "w") as f:
                json.dump(new_judge_result.to_dict(), f, indent=2)

        return result

    async def _call_llm(self, original_decision: str, critique: str,
                        improvements: list[str], score: float, verdict: str) -> str:
        """Single LLM call to revise the decision with retry on 503."""
        base_url = _get_base_url()
        api_key = _get_api_key()
        model = _get_model()

        if not api_key:
            raise RuntimeError("OPENCODE_GO_API_KEY not set")

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        user_prompt = REFINE_USER_PROMPT.format(
            original_decision=original_decision,
            critique=critique,
            improvement_1=improvements[0] if len(improvements) > 0 else "N/A",
            improvement_2=improvements[1] if len(improvements) > 1 else "N/A",
            improvement_3=improvements[2] if len(improvements) > 2 else "N/A",
            score=score,
            verdict=verdict,
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.3,
        }

        from _retry import httpx_retry

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await httpx_retry(client, "POST", url, headers=headers, json=payload)
            if resp.status_code == 429:
                raise RuntimeError("OpenCode Go rate limit exhausted")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def save_result(self, result: RefineResult) -> Path:
        """Save refinement result."""
        results_dir = _get_results_dir()
        out_dir = results_dir / result.ticker.upper() / "TradingAgentsStrategy_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"refine_result_{result.date}.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return out_path


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------

def run_refine(ticker: str, date: str, save: bool = True, force: bool = False) -> RefineResult:
    """Synchronous entry point."""
    import asyncio

    async def _run():
        refiner = Refiner()
        result = await refiner.refine(ticker, date, force=force)
        if save and not result.error:
            path = refiner.save_result(result)
            print(f"Saved: {path}", flush=True)
        return result

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    if len(sys.argv) < 3:
        print("Usage: python judge_refine.py TICKER DATE [--force]")
        print("  python judge_refine.py NVDA 2026-05-01")
        print("  python judge_refine.py AAPL 2026-05-01 --force  # override threshold")
        print("  Requires: judge score must already exist for this ticker+date")
        sys.exit(1)

    ticker = sys.argv[1]
    date = sys.argv[2]
    force = "--force" in sys.argv

    print(f"Refining {ticker} ({date}){' [FORCE]' if force else ''}...", flush=True)
    print(f"Step 1: Generating revised decision...", flush=True)
    result = run_refine(ticker, date, force=force)

    if result.error:
        print(f"\nError: {result.error}")
        sys.exit(1)

    print(f"\n{result.to_telegram()}")
