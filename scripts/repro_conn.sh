#!/usr/bin/env bash
cd /opt/data/workspace/TradingAgents
.venv/bin/python scripts/run_analysis.py --help 2>&1 | grep -E "deep-llm|quick-llm"
