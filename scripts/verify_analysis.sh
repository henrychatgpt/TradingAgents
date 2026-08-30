#!/usr/bin/env bash
# Verify the full TRTA analysis pipeline end-to-end.
cd /opt/data/workspace/TradingAgents
.venv/bin/python scripts/run_analysis.py --ticker SKHY --date 2026-08-07 > logs/verify_analysis.log 2>&1
echo "EXIT=$?" >> logs/verify_analysis.log
