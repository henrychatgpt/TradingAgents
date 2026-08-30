#!/usr/bin/env bash
# Direct connectivity test to the LLM provider used by TradingAgents.
cd /opt/data/workspace/TradingAgents
KEY=$(sed -n 's/^OPENCODE_GO_API_KEY=//p' .env)
echo "== opencode-go test =="
curl -s --max-time 25 https://opencode.ai/zen/go/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"reply with the single word ok"}],"max_tokens":10}' \
  | head -c 300
echo
echo "== deepseek fallback test =="
DK=$(sed -n 's/^DEEPSEEK_API_KEY=//p' .env)
curl -s --max-time 25 https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer $DK" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"reply with the single word ok"}],"max_tokens":10}' \
  | head -c 300
echo
