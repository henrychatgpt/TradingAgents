"""
Judge Calibration Module — Phase 2
====================================
Correlates LLM-as-Judge quality scores with actual market outcomes.

For each analysis:
1. Extracts trade parameters (recommendation, entry, stop, target) from full_states_log
2. Fetches post-analysis price data via yfinance
3. Computes outcome metrics (P&L, stop hit, target hit, max drawdown/favorable)
4. Correlates with judge scores to find which dimensions predict good outcomes

Usage:
    from judge_calibration import JudgeCalibration
    cal = JudgeCalibration()
    report = cal.run()                    # Calibrate all available analyses
    report = cal.run(ticker="TSLA")       # Calibrate single ticker
    report = cal.run(date="2026-05-01")   # Calibrate single date

    # CLI:
    python judge_calibration.py
    python judge_calibration.py TSLA 2026-05-01
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _get_results_dir():
    return Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "/opt/data/workspace/TradingAgents/results"))

HKT = timezone(timedelta(hours=8))

# Recommendation → expected direction
DIRECTION_MAP = {
    "overweight": "bullish",
    "buy": "bullish",
    "strong buy": "bullish",
    "hold": "neutral",
    "underweight": "bearish",
    "sell": "bearish",
    "strong sell": "bearish",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeParams:
    """Extracted trade parameters from an analysis."""
    ticker: str
    date: str
    recommendation: str = ""       # e.g. "Overweight", "Hold", "Sell"
    direction: str = ""            # bullish / neutral / bearish
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    raw_text: str = ""             # First 500 chars of final_trade_decision

@dataclass
class MarketOutcome:
    """Actual market outcome after the analysis."""
    ticker: str
    analysis_date: str
    entry_price: float             # Close on analysis date
    current_price: float           # Latest available close
    days_held: int = 0
    pnl_pct: float = 0.0          # Simple return from entry to current
    max_favorable_pct: float = 0.0 # Max upside seen (high - entry)
    max_adverse_pct: float = 0.0   # Max downside seen (entry - low)
    stop_hit: bool = False
    target_hit: bool = False
    direction_correct: Optional[bool] = None  # Did price move as predicted?

@dataclass
class CalibrationRow:
    """One analysis row: judge score + market outcome."""
    ticker: str
    date: str
    recommendation: str
    direction: str
    judge_score: float
    judge_verdict: str
    dimensions: dict              # name → score
    entry_price: Optional[float]
    current_price: float
    pnl_pct: float
    max_favorable_pct: float
    max_adverse_pct: float
    stop_hit: bool
    target_hit: bool
    direction_correct: Optional[bool]

@dataclass
class CalibrationReport:
    """Full calibration report."""
    generated_at: str = ""
    total_analyses: int = 0
    rows: list[CalibrationRow] = field(default_factory=list)
    # Aggregate stats
    avg_judge_score_correct: float = 0.0
    avg_judge_score_wrong: float = 0.0
    dimension_correlations: dict = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_analyses": self.total_analyses,
            "rows": [asdict(r) for r in self.rows],
            "avg_judge_score_correct": self.avg_judge_score_correct,
            "avg_judge_score_wrong": self.avg_judge_score_wrong,
            "dimension_correlations": self.dimension_correlations,
            "insights": self.insights,
        }

    def to_telegram(self) -> str:
        """Plain-text calibration report for Telegram."""
        lines = [
            "Judge Calibration Report",
            f"Generated: {self.generated_at}",
            f"Analyses: {self.total_analyses}",
            "",
        ]

        # Per-row summary
        lines.append("Individual Results:")
        for r in self.rows:
            # numpy.bool_ doesn't pass `is True` — use == comparison
            dc = bool(r.direction_correct) if r.direction_correct is not None else None
            direction_icon = "?"
            if dc is True:
                direction_icon = "Y"
            elif dc is False:
                direction_icon = "N"

            pnl_str = f"{r.pnl_pct:+.2f}%"
            judge_str = f"{r.judge_score:.1f}"
            fav_str = f"+{r.max_favorable_pct:.1f}%"
            adv_str = f"-{r.max_adverse_pct:.1f}%"

            lines.append(
                f"  {r.ticker} ({r.recommendation}) "
                f"Judge={judge_str} PnL={pnl_str} "
                f"Range=[{adv_str},{fav_str}] Dir={direction_icon}"
            )

        # Aggregate
        if self.avg_judge_score_correct > 0 or self.avg_judge_score_wrong > 0:
            lines.append("")
            lines.append("Score vs Outcome Correlation:")
            if self.avg_judge_score_correct > 0:
                lines.append(f"  Avg judge score (direction correct): {self.avg_judge_score_correct:.1f}")
            if self.avg_judge_score_wrong > 0:
                lines.append(f"  Avg judge score (direction wrong):   {self.avg_judge_score_wrong:.1f}")

        # Dimension correlations
        if self.dimension_correlations:
            lines.append("")
            lines.append("Dimension Predictiveness:")
            for dim, corr in sorted(self.dimension_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                arrow = "+" if corr > 0 else ""
                lines.append(f"  {dim}: {arrow}{corr:.2f}")

        # Insights
        if self.insights:
            lines.append("")
            lines.append("Insights:")
            for i, ins in enumerate(self.insights, 1):
                lines.append(f"  {i}. {ins}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trade parameter extraction
# ---------------------------------------------------------------------------

def extract_trade_params(data: dict) -> TradeParams:
    """Extract trade parameters from a full_states_log JSON."""
    ticker = data.get("company_of_interest", "?")
    date = data.get("trade_date", "?")
    final_text = data.get("final_trade_decision", "")

    params = TradeParams(
        ticker=ticker,
        date=date,
        raw_text=final_text[:500],
    )

    # Extract recommendation (Rating line)
    rating_match = re.search(r"\*\*Rating\*\*:\s*(.+?)[\n\r]", final_text)
    if rating_match:
        params.recommendation = rating_match.group(1).strip()
        params.direction = DIRECTION_MAP.get(params.recommendation.lower(), "neutral")

    # Extract stop loss — multiple patterns
    stop_patterns = [
        r"[Ss]top[\s-]*(?:[Ll]oss)?[:\s]+\$?([\d,]+\.?\d*)",
        r"[Ss]top(?:\s+at)?[:\s]+\$?([\d,]+\.?\d*)",
        r"[Pp]rotective stop at \$?([\d,]+\.?\d*)",
        r"[Hh]ard stop at \$?([\d,]+\.?\d*)",
    ]
    for pat in stop_patterns:
        m = re.search(pat, final_text)
        if m:
            params.stop_loss = float(m.group(1).replace(",", ""))
            break

    # Extract target — multiple patterns
    target_patterns = [
        r"[Tt]arget(?:\s+(?:the|at|price))?[:\s]+\$?([\d,]+\.?\d*)",
        r"[Tt]argeting[:\s]+\$?([\d,]+\.?\d*)",
        r"[Uu]pside target[:\s]+\$?([\d,]+\.?\d*)",
    ]
    for pat in target_patterns:
        m = re.search(pat, final_text)
        if m:
            val = float(m.group(1).replace(",", ""))
            # Sanity: target should be different from stop
            if params.stop_loss and abs(val - params.stop_loss) < 1:
                continue
            params.target = val
            break

    # Extract entry price
    entry_patterns = [
        r"(?:at|entry at|enter at)[:\s]+\$?([\d,]+\.?\d*)",
        r"(?:current|price)[:\s]+\(?~?\$?([\d,]+\.?\d*)\)?",
    ]
    for pat in entry_patterns:
        m = re.search(pat, final_text, re.IGNORECASE)
        if m:
            params.entry_price = float(m.group(1).replace(",", ""))
            break

    return params


# ---------------------------------------------------------------------------
# Market outcome computation
# ---------------------------------------------------------------------------

def compute_outcome(params: TradeParams, days_to_check: int = 30) -> MarketOutcome:
    """Fetch post-analysis price data and compute outcome metrics.
    
    Uses the analysis date's close as the reference entry price.
    Uses intraday high/low for max favorable/adverse excursion.
    Compares analysis-day close vs PREVIOUS-day close to measure
    same-day and multi-day accuracy.
    """
    ticker = params.ticker
    analysis_date = params.date

    outcome = MarketOutcome(
        ticker=ticker,
        analysis_date=analysis_date,
        entry_price=0,
        current_price=0,
    )

    try:
        tk = yf.Ticker(ticker)
        # Get history from 5 days before analysis through today
        # to have pre-analysis reference price
        from datetime import datetime as dt
        a_date = dt.strptime(analysis_date, "%Y-%m-%d")
        from datetime import timedelta as td
        start_date = (a_date - td(days=7)).strftime("%Y-%m-%d")
        
        hist = tk.history(start=start_date, period="1mo")
        if hist.empty:
            outcome.current_price = 0
            return outcome

        # Find the analysis date row
        analysis_idx = None
        for i, idx in enumerate(hist.index):
            if idx.strftime("%Y-%m-%d") == analysis_date:
                analysis_idx = i
                break

        if analysis_idx is None:
            # Fallback: use first available row
            analysis_idx = 0

        # Entry = close on analysis date (what the analysts saw)
        entry = hist["Close"].iloc[analysis_idx]
        outcome.entry_price = round(entry, 2)

        # Current = latest available close
        current = hist["Close"].iloc[-1]
        outcome.current_price = round(current, 2)

        # Days held = trading days after analysis date
        outcome.days_held = max(0, len(hist) - 1 - analysis_idx)

        if entry > 0:
            # PnL from analysis close to current close
            outcome.pnl_pct = round((current - entry) / entry * 100, 2)

            # Use intraday range on analysis day AND subsequent days
            post_hist = hist.iloc[analysis_idx:]  # from analysis day onwards
            max_high = post_hist["High"].max()
            min_low = post_hist["Low"].min()

            outcome.max_favorable_pct = round((max_high - entry) / entry * 100, 2)
            outcome.max_adverse_pct = round((entry - min_low) / entry * 100, 2)

        # Check stop/target hit using full range from analysis day
        post_hist = hist.iloc[analysis_idx:]
        
        if params.stop_loss:
            if params.direction in ("bearish",):
                # Bearish stop is above entry — hit if high reaches stop
                if post_hist["High"].max() >= params.stop_loss:
                    outcome.stop_hit = True
            else:
                # Bullish/neutral stop is below entry — hit if low reaches stop
                if post_hist["Low"].min() <= params.stop_loss:
                    outcome.stop_hit = True

        if params.target:
            if params.direction in ("bearish",):
                if post_hist["Low"].min() <= params.target:
                    outcome.target_hit = True
            else:
                if post_hist["High"].max() >= params.target:
                    outcome.target_hit = True

        # Direction correctness: compare analysis close vs PREVIOUS close
        # Did the price move in the predicted direction after analysis?
        if analysis_idx > 0:
            prev_close = hist["Close"].iloc[analysis_idx - 1]
            actual_move = (entry - prev_close) / prev_close * 100
        else:
            actual_move = 0

        if params.direction == "bullish":
            # Bullish call correct if current price > analysis close
            outcome.direction_correct = current > entry
        elif params.direction == "bearish":
            # Bearish call correct if current price < analysis close
            outcome.direction_correct = current < entry
        elif params.direction == "neutral":
            # Hold correct if didn't move more than 5%
            outcome.direction_correct = abs(outcome.pnl_pct) < 5.0
        else:
            outcome.direction_correct = None

    except Exception as e:
        outcome.current_price = 0

    return outcome


# ---------------------------------------------------------------------------
# Calibration engine
# ---------------------------------------------------------------------------

class JudgeCalibration:
    """Correlate judge scores with market outcomes."""

    def __init__(self):
        self.results_dir = _get_results_dir()

    def find_analyses(self, ticker: str = None, date: str = None) -> list[dict]:
        """Find all analysis logs with matching judge scores."""
        analyses = []
        results_dir = self.results_dir

        if not results_dir.exists():
            return analyses

        # Determine which tickers to scan
        if ticker:
            ticker_dirs = [results_dir / ticker.upper()]
        else:
            ticker_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

        for tdir in ticker_dirs:
            if not tdir.is_dir():
                continue
            logs_dir = tdir / "TradingAgentsStrategy_logs"
            if not logs_dir.exists():
                continue

            # Find matching full_states_log files
            if date:
                log_files = [logs_dir / f"full_states_log_{date}.json"]
            else:
                log_files = sorted(logs_dir.glob("full_states_log_*.json"))

            for log_file in log_files:
                if not log_file.exists():
                    continue
                # Check if judge score exists
                date_str = log_file.stem.replace("full_states_log_", "")
                judge_file = logs_dir / f"judge_score_{date_str}.json"
                if not judge_file.exists():
                    continue  # Skip analyses without judge scores

                analyses.append({
                    "ticker": tdir.name,
                    "date": date_str,
                    "log_file": log_file,
                    "judge_file": judge_file,
                })

        return analyses

    def run(self, ticker: str = None, date: str = None) -> CalibrationReport:
        """Run calibration on all (or filtered) analyses."""
        report = CalibrationReport(
            generated_at=datetime.now(HKT).strftime("%Y-%m-%d %H:%M HKT")
        )

        analyses = self.find_analyses(ticker, date)
        if not analyses:
            report.error = "No analyses with judge scores found"
            return report

        report.total_analyses = len(analyses)

        for a in analyses:
            try:
                # Load analysis log
                with open(a["log_file"]) as f:
                    log_data = json.load(f)

                # Load judge score
                with open(a["judge_file"]) as f:
                    judge_data = json.load(f)

                # Extract trade params
                params = extract_trade_params(log_data)

                # Compute market outcome
                outcome = compute_outcome(params)

                if outcome.current_price == 0:
                    continue  # Skip if no price data

                # Build dimension dict
                dim_scores = {}
                for d in judge_data.get("dimensions", []):
                    dim_scores[d["name"]] = d["score"]

                row = CalibrationRow(
                    ticker=a["ticker"],
                    date=a["date"],
                    recommendation=params.recommendation,
                    direction=params.direction,
                    judge_score=judge_data.get("overall_score", 0),
                    judge_verdict=judge_data.get("verdict", "?"),
                    dimensions=dim_scores,
                    entry_price=outcome.entry_price,
                    current_price=outcome.current_price,
                    pnl_pct=outcome.pnl_pct,
                    max_favorable_pct=outcome.max_favorable_pct,
                    max_adverse_pct=outcome.max_adverse_pct,
                    stop_hit=outcome.stop_hit,
                    target_hit=outcome.target_hit,
                    direction_correct=outcome.direction_correct,
                )
                report.rows.append(row)

            except Exception as e:
                # Skip problematic analyses
                continue

        # Compute aggregate stats
        self._compute_aggregates(report)

        return report

    def _compute_aggregates(self, report: CalibrationReport):
        """Compute correlation between judge scores and outcomes."""
        rows = report.rows
        if not rows:
            return

        # Separate correct vs incorrect direction predictions
        correct = [r for r in rows if bool(r.direction_correct) is True]
        wrong = [r for r in rows if bool(r.direction_correct) is False]

        if correct:
            report.avg_judge_score_correct = round(
                sum(r.judge_score for r in correct) / len(correct), 1
            )
        if wrong:
            report.avg_judge_score_wrong = round(
                sum(r.judge_score for r in wrong) / len(wrong), 1
            )

        # Dimension-level correlation with P&L
        all_dims = set()
        for r in rows:
            all_dims.update(r.dimensions.keys())

        for dim in all_dims:
            pairs = [(r.dimensions.get(dim, 0), r.pnl_pct) for r in rows if dim in r.dimensions]
            if len(pairs) >= 2:
                corr = self._pearson_r([p[0] for p in pairs], [p[1] for p in pairs])
                report.dimension_correlations[dim] = round(corr, 2)

        # Generate insights
        report.insights = self._generate_insights(report, correct, wrong)

    @staticmethod
    def _pearson_r(x: list[float], y: list[float]) -> float:
        """Simple Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)

    @staticmethod
    def _generate_insights(report: CalibrationReport, correct: list, wrong: list) -> list[str]:
        """Generate human-readable insights from calibration data."""
        insights = []
        rows = report.rows

        if not rows:
            return insights

        # Insight 1: Direction accuracy
        total_with_dir = len(correct) + len(wrong)
        if total_with_dir > 0:
            acc = len(correct) / total_with_dir * 100
            insights.append(
                f"Direction accuracy: {acc:.0f}% ({len(correct)}/{total_with_dir}) "
                f"across {len(rows)} analyses"
            )

        # Insight 2: Judge score gap
        if correct and wrong:
            gap = report.avg_judge_score_correct - report.avg_judge_score_wrong
            if abs(gap) >= 0.5:
                direction = "higher" if gap > 0 else "lower"
                insights.append(
                    f"Correct predictions had {direction} judge scores "
                    f"(gap: {gap:+.1f} points) — judge has {'some' if abs(gap) < 2 else 'strong'} predictive value"
                )
            else:
                insights.append(
                    f"Minimal score gap ({gap:+.1f}) between correct/wrong predictions — "
                    f"judge quality score does NOT predict market direction yet"
                )

        # Insight 3: Best/worst dimension
        if report.dimension_correlations:
            best_dim = max(report.dimension_correlations, key=lambda k: abs(report.dimension_correlations[k]))
            best_corr = report.dimension_correlations[best_dim]
            insights.append(
                f"Most predictive dimension: {best_dim} (r={best_corr:+.2f})"
            )

        # Insight 4: Stop/target reliability
        stop_rows = [r for r in rows if r.stop_hit]
        target_rows = [r for r in rows if r.target_hit]
        if stop_rows or target_rows:
            insights.append(
                f"Stops hit: {len(stop_rows)}/{len(rows)}, Targets hit: {len(target_rows)}/{len(rows)}"
            )

        # Insight 5: Sample size warning
        if len(rows) < 10:
            insights.append(
                f"Small sample ({len(rows)} analyses) — correlations are NOT statistically significant. "
                f"Need 30+ analyses for reliable calibration"
            )

        # Insight 6: P&L distribution
        pnls = [r.pnl_pct for r in rows]
        avg_pnl = sum(pnls) / len(pnls)
        best = max(rows, key=lambda r: r.pnl_pct)
        worst = min(rows, key=lambda r: r.pnl_pct)
        insights.append(
            f"Avg P&L: {avg_pnl:+.2f}%, Best: {best.ticker} {best.pnl_pct:+.2f}%, "
            f"Worst: {worst.ticker} {worst.pnl_pct:+.2f}%"
        )

        return insights

    def save_report(self, report: CalibrationReport, path: Path = None) -> Path:
        """Save calibration report to JSON."""
        if path is None:
            path = self.results_dir / "calibration_reports" / f"calibration_{report.generated_at.replace(':', '').replace(' ', '_')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    cal = JudgeCalibration()

    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    date = sys.argv[2] if len(sys.argv) > 2 else None

    print("Running calibration...", flush=True)
    report = cal.run(ticker=ticker, date=date)

    if report.error:
        print(f"Error: {report.error}")
        sys.exit(1)

    print(report.to_telegram())

    # Save
    path = cal.save_report(report)
    print(f"\nSaved: {path}")
