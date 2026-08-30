#!/usr/bin/env bash
# Start TRTA bot fully detached — survives background process wrapper issues
# The Hermes background wrapper kills asyncio event loops silently.
# This script double-forks to fully detach from the parent process group.

set -e
cd /opt/data/workspace/TradingAgents

export TRADINGAGENTS_BOT_TOKEN="8529551365:AAGbrlhOI17-stm5R71bi_kPggQxCCDTw3o"
export PYTHONUNBUFFERED=1
LOGFILE="/opt/data/workspace/TradingAgents/logs/ta_bot.log"
mkdir -p logs

# Kill any existing instance
pkill -f "ta_telegram_bot" 2>/dev/null || true
sleep 2

# Double-fork to fully detach
(
    (
        exec .venv/bin/python -u scripts/ta_telegram_bot.py >> "$LOGFILE" 2>&1
    ) &
)
echo "Bot started. Log: $LOGFILE"
