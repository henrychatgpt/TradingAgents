#!/usr/bin/env python3
"""Test the discussion Q&A path with the new deepseek fallback."""
import asyncio
import os
import sys

sys.path.insert(0, "/opt/data/workspace/TradingAgents/scripts")
os.chdir("/opt/data/workspace/TradingAgents")
from dotenv import load_dotenv
load_dotenv("/opt/data/workspace/TradingAgents/.env", override=True)

from discussion import opencode_go_chat


async def main():
    print("calling opencode_go_chat (primary exhausted -> should fall back)...", flush=True)
    r = await opencode_go_chat(
        [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=50)
    print("RESULT:", r[:120], flush=True)


asyncio.run(main())
