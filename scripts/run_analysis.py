#!/usr/bin/env python3
"""TradingAgents runner script — one-liner to run analysis from CLI."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# Load env before importing tradingagents
from dotenv import load_dotenv
load_dotenv("/opt/data/workspace/TradingAgents/.env")

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="TradingAgents Analysis Runner")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL, NVDA)")
    parser.add_argument("--date", default=None, help="Analysis date YYYY-MM-DD (default: last trading day)")
    parser.add_argument("--output-language", default="English", help="Report language")
    parser.add_argument("--checkpoint", action="store_true", help="Enable checkpoint resume")
    parser.add_argument("--debate-rounds", type=int, default=1, help="Bull/Bear debate rounds")
    parser.add_argument("--risk-rounds", type=int, default=1, help="Risk debate rounds")
    parser.add_argument("--deep-llm", default="glm-5.3-flash", help="Deep thinking model (default: glm-5.3-flash; kimi-k3 is reserved for vision tasks)")
    parser.add_argument("--quick-llm", default="glm-5.3-flash", help="Quick thinking model (default: glm-5.3-flash; kimi-k3 is reserved for vision tasks)")
    args = parser.parse_args()

    # Auto-detect last trading day if --date not provided
    if args.date is None:
        from datetime import datetime, timedelta
        import pandas as pd
        today = datetime.now()
        # Go back to find last business day (skip weekends)
        for i in range(1, 7):
            candidate = today - timedelta(days=i)
            if candidate.weekday() < 5:  # Mon-Fri
                # Check if market was open (not a US holiday)
                try:
                    cal = pd.tseries.offsets.CustomBusinessDay(calendar=pd.tseries.holiday.USFederalHolidayCalendar())
                    prev_bday = (today - cal).strftime("%Y-%m-%d")
                    args.date = prev_bday
                except Exception:
                    args.date = candidate.strftime("%Y-%m-%d")
                break
        log(f"No --date provided, using last trading day: {args.date}")

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "opencode-go"
    config["deep_think_llm"] = args.deep_llm
    config["quick_think_llm"] = args.quick_llm
    config["results_dir"] = "/opt/data/workspace/TradingAgents/results"
    config["data_cache_dir"] = "/opt/data/workspace/TradingAgents/cache"
    config["memory_log_path"] = "/opt/data/workspace/TradingAgents/memory/trading_memory.md"
    config["output_language"] = args.output_language
    config["max_debate_rounds"] = args.debate_rounds
    config["max_risk_discuss_rounds"] = args.risk_rounds
    # Enable checkpoint by default for fallback resilience
    config["checkpoint_enabled"] = True

    log(f"Analyzing {args.ticker} on {args.date}...")
    log(f"Primary provider: opencode-go | Deep: {args.deep_llm} | Quick: {args.quick_llm}")
    log(f"Fallback: deepseek | Model: deepseek-chat")
    log(f"Debates: {args.debate_rounds} invest, {args.risk_rounds} risk")

    # Try primary provider first, fall back to deepseek on provider errors
    decision = None
    primary_provider = "opencode-go"
    primary_deep = args.deep_llm
    primary_quick = args.quick_llm
    fallback_provider = "deepseek"
    fallback_deep = "deepseek-chat"
    fallback_quick = "deepseek-chat"

    for attempt, (provider, deep, quick) in enumerate([
        (primary_provider, primary_deep, primary_quick),
        (fallback_provider, fallback_deep, fallback_quick),
    ], 1):
        try:
            config["llm_provider"] = provider
            config["deep_think_llm"] = deep
            config["quick_think_llm"] = quick

            log(f"Attempt {attempt}: {provider} ({deep})")
            ta = TradingAgentsGraph(debug=True, config=config)
            _, decision = ta.propagate(args.ticker, args.date)
            break  # success
        except Exception as e:
            err_str = str(e)
            log(f"Attempt {attempt} failed: {type(e).__name__}: {err_str[:200]}")
            if attempt == 1 and (
                "503" in err_str or "429" in err_str
                or "Error code: 500" in err_str or "Error code: 502" in err_str
                or "Error code: 504" in err_str
                or "Internal server error" in err_str
                or "failover_exhausted" in err_str
                or "GoUsageLimitError" in err_str
                or "Inference is temporarily unavailable" in err_str
                or "Too Many Requests" in err_str
                or "Connection error" in err_str
                or "APIConnectionError" in err_str
            ):
                log("Provider error — falling back to deepseek...")
                continue  # try fallback
            else:
                raise  # not a provider error, re-raise

    if decision is None:
        log("All attempts failed. No decision produced.")
        sys.exit(1)

    # Save quick result
    result_path = f"/opt/data/workspace/TradingAgents/results/{args.ticker.lower()}_{args.date}.json"
    with open(result_path, "w") as f:
        json.dump({"ticker": args.ticker, "date": args.date, "decision": decision}, f, indent=2)

    log(f"Decision: {decision}")
    log(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
