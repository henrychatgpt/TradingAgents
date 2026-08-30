#!/usr/bin/env python3
"""Check which opencode-go models still respond (kimi-k3 vs deepseek-v4-flash)."""
import os
import sys

sys.path.insert(0, "/opt/data/workspace/TradingAgents")
os.chdir("/opt/data/workspace/TradingAgents")
from dotenv import load_dotenv
load_dotenv("/opt/data/workspace/TradingAgents/.env", override=True)

from openai import OpenAI

c = OpenAI(api_key=os.environ["OPENCODE_GO_API_KEY"], base_url="https://opencode.ai/zen/go/v1")

for model in ["kimi-k3", "deepseek-v4-flash"]:
    try:
        r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "say ok"}], max_tokens=50)
        print(f"{model}: OK ->", repr((r.choices[0].message.content or "")[:30]))
    except Exception as e:
        print(f"{model}: FAIL", type(e).__name__, str(e)[:140])
