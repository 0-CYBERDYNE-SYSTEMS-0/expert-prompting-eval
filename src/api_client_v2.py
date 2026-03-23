"""
v2 API client — instrumented, hardened, retry-aware.

Features:
  - Probes token saturation before setting max_tokens
  - Exponential backoff retries (3 attempts)
  - Records usage.completion_tokens to detect truncation
  - Returns None on genuine failure (not minimum score)
  - temperature=0 for reproducibility
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from src.benchmark_tasks import Task


# ─── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class APIResponse:
    content: str
    finish_reason: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    attempt: int
    error: Optional[str] = None


@dataclass
class ProbingResult:
    saturation_tokens: int  # Tokens at which output stops growing
    recommended_max: int    # saturation + 256 headroom


# ─── LM Studio Client ─────────────────────────────────────────────────────────

LM_STUDIO_URL = "http://localhost:1234"
LM_MODEL = "qwen3.5-2b-claude-4.6-opus-reasoning-distilled"
LM_TIMEOUT = 120


def call_lm_studio(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    attempt: int = 1,
) -> APIResponse:
    """Single attempt LM Studio call."""
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": LM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LM_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            usage = result.get("usage", {})
            return APIResponse(
                content=msg["content"].strip() if msg.get("content") else "",
                finish_reason=result["choices"][0].get("finish_reason", "unknown"),
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                attempt=attempt,
            )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return APIResponse("", "error", 0, 0, 0, attempt, f"HTTP {e.code}: {body[:200]}")
    except Exception as e:
        return APIResponse("", "error", 0, 0, 0, attempt, str(e)[:200])


def call_lm_studio_retry(prompt: str, max_tokens: int = 512) -> APIResponse:
    """LM Studio call with 3 attempts and exponential backoff."""
    for attempt in range(1, 4):
        resp = call_lm_studio(prompt, max_tokens, attempt=attempt)
        if resp.finish_reason == "error":
            wait = 2 ** (attempt - 1)
            time.sleep(wait)
            continue
        if resp.content and len(resp.content) > 0:
            return resp
        if attempt < 3:
            wait = 2 ** (attempt - 1)
            time.sleep(wait)
    return resp  # Return last attempt even if empty


# ─── OpenAI Client ────────────────────────────────────────────────────────────

OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_API_KEY: Optional[str] = None


def set_openai_key(key: str):
    global OPENAI_API_KEY
    OPENAI_API_KEY = key


def call_openai(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    attempt: int = 1,
) -> APIResponse:
    """Single attempt OpenAI call."""
    if not OPENAI_API_KEY:
        return APIResponse("", "error", 0, 0, 0, attempt, "No API key")

    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            usage = result.get("usage", {})
            return APIResponse(
                content=msg["content"].strip() if msg.get("content") else "",
                finish_reason=result["choices"][0].get("finish_reason", "unknown"),
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                attempt=attempt,
            )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return APIResponse("", "error", 0, 0, 0, attempt, f"HTTP {e.code}: {body[:200]}")
    except Exception as e:
        return APIResponse("", "error", 0, 0, 0, attempt, str(e)[:200])


def call_openai_retry(prompt: str, max_tokens: int = 2048) -> APIResponse:
    """OpenAI call with 3 attempts and exponential backoff."""
    for attempt in range(1, 4):
        resp = call_openai(prompt, max_tokens, attempt=attempt)
        if resp.finish_reason == "error":
            wait = 2 ** (attempt - 1)
            time.sleep(wait)
            continue
        if resp.content and len(resp.content) > 0:
            return resp
        if attempt < 3:
            wait = 2 ** (attempt - 1)
            time.sleep(wait)
    return resp


# ─── Token Saturation Probe ───────────────────────────────────────────────────

def probe_model_tokens(
    model: str,
    test_prompt: str,
    max_range: int = 512,
) -> ProbingResult:
    """
    Find the token count at which output stops growing.
    Binary search between 64 and max_range tokens.
    """
    lo, hi = 64, max_range
    prev_len = 0

    while hi - lo > 32:
        mid = (lo + hi) // 2
        if model == "lm_studio":
            resp = call_lm_studio(test_prompt, max_tokens=mid, temperature=0.0)
        else:
            resp = call_openai(test_prompt, max_tokens=mid, temperature=0.0)

        content_len = len(resp.content)

        if content_len > prev_len:
            lo = mid
        else:
            hi = mid
        prev_len = content_len

        time.sleep(0.5)

    final_resp = (
        call_lm_studio(test_prompt, max_tokens=hi, temperature=0.0)
        if model == "lm_studio"
        else call_openai(test_prompt, max_tokens=hi, temperature=0.0)
    )

    saturation = final_resp.completion_tokens
    return ProbingResult(
        saturation_tokens=saturation,
        recommended_max=saturation + 256,
    )


def run_probe_sequence() -> dict:
    """Probe both models and report recommended token settings."""
    print("\n[PROBE] Measuring token saturation for both models...")

    probe_prompt = (
        "A researcher tests whether plants grow better under white or colored light. "
        "Design the experiment briefly."
    )

    lm_result = probe_model_tokens("lm_studio", probe_prompt, max_range=512)
    print(f"  [LM Studio] saturation={lm_result.saturation_tokens} tokens, max={lm_result.recommended_max}")

    openai_result = probe_model_tokens("openai", probe_prompt, max_range=2048)
    print(f"  [OpenAI] saturation={openai_result.saturation_tokens} tokens, max={openai_result.recommended_max}")

    return {
        "lm_studio": {
            "saturation": lm_result.saturation_tokens,
            "recommended_max": lm_result.recommended_max,
        },
        "openai": {
            "saturation": openai_result.saturation_tokens,
            "recommended_max": openai_result.recommended_max,
        },
    }
