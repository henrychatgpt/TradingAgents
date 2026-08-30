#!/usr/bin/env bash
# Restart TRTA bot and verify it's up and polling.
cd /opt/data/workspace/TradingAgents
bash scripts/start_bot.sh
sleep 10
echo "==== process ===="
ps aux | grep "[t]elegram_bot" | awk '{print "pid", $2, $9, $11, $12}' || echo "NOT RUNNING"
echo "==== log tail ===="
tail -6 logs/ta_bot.log
