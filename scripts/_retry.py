"""
Retry utilities for OpenCode Go API calls.

OpenCode Go occasionally returns 503 (temporarily unavailable) or 429
(rate limited). This module provides retry wrappers with exponential
backoff for both httpx (direct) and langchain (ChatOpenAI) call sites.
"""

import asyncio
import time
from typing import Optional

import httpx


class RetryExhausted(Exception):
    """Raised when all retry attempts fail."""


async def httpx_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[dict] = None,
    json: Optional[dict] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> httpx.Response:
    """
    Send an HTTP request with exponential backoff on 503/429/5xx.

    Retries on:
      - 503 (service unavailable)
      - 429 (rate limited, respecting Retry-After header)
      - Any 5xx server error

    Raises ``RetryExhausted`` after ``max_retries`` failed attempts.
    On the final attempt the last exception propagates directly.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.request(method, url, headers=headers, json=json)

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp, default=base_delay * 2 ** (attempt - 1))
                if attempt < max_retries:
                    delay = min(retry_after, 30)
                    print(f"[RETRY] 429 on {url} — retry {attempt}/{max_retries} after {delay:.0f}s", flush=True)
                    await asyncio.sleep(delay)
                    continue
                return resp  # Last attempt — let caller handle it

            if resp.status_code in (503, 502, 504) or (500 <= resp.status_code < 600):
                if attempt < max_retries:
                    delay = min(base_delay * 2 ** (attempt - 1), 30)
                    print(f"[RETRY] {resp.status_code} on {url} — retry {attempt}/{max_retries} after {delay:.0f}s", flush=True)
                    await asyncio.sleep(delay)
                    continue
                return resp  # Last attempt

            return resp  # Success (2xx, 4xx other than 429)

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(base_delay * 2 ** (attempt - 1), 30)
                print(f"[RETRY] Network error on {url} — retry {attempt}/{max_retries} after {delay:.0f}s: {e}", flush=True)
                await asyncio.sleep(delay)
                continue
            raise RetryExhausted(f"Network error after {max_retries} attempts: {e}") from e

    # Shouldn't reach here, but just in case
    if last_exc:
        raise last_exc
    raise RetryExhausted(f"Failed after {max_retries} attempts")


def _parse_retry_after(resp: httpx.Response, default: float) -> float:
    """Extract Retry-After header value in seconds."""
    val = resp.headers.get("Retry-After")
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default
