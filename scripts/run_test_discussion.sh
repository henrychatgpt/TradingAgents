#!/usr/bin/env bash
cd /opt/data/workspace/TradingAgents
.venv/bin/python scripts/test_discussion_fb.py 2>&1 | tail -6
