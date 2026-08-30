#!/usr/bin/env bash
# Restart the TradingAgents (TRTA) Telegram bot daemon.
cd /opt/data/workspace/TradingAgents/scripts || exit 1
pkill -f "ta_telegram_bot" 2>/dev/null
sleep 2
nohup /opt/data/workspace/TradingAgents/.venv/bin/python -u ta_telegram_bot.py >> ../logs/ta_bot.log 2>&1 &
echo "trta pid $!"
sleep 8
TOK=$(grep "^TRADINGAGENTS_BOT_TOKEN=" ../.env | cut -d= -f2-)
curl -s --max-time 15 "https://api.telegram.org/bot${TOK}/getUpdates?timeout=0" | head -c 120
echo ""
tail -4 ../logs/ta_bot.log
