#!/usr/bin/env bash
# Reinstall tradingagents as editable with the CURRENT path, verify import.
set -e
cd /opt/data/workspace/TradingAgents
uv pip install --python .venv/bin/python -e . 2>&1 | tail -3
.venv/bin/python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph; print('tradingagents import OK')"
