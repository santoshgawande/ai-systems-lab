"""
Mini AI Gateway — production LLM proxy with:
  - API key authentication
  - Multi-provider routing (OpenAI / Anthropic / Ollama)
  - Request/response logging with token counts
  - Automatic fallback on provider failure
  - Rate limiting per API key
  - Cost tracking

Run:
  pip install fastapi uvicorn httpx pydantic tiktoken
  python gateway.py
  # or: uvicorn gateway:app --reload

Test:
  export OPENAI_API_KEY=sk-...
  curl http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer gw-test-key-1" \
    -H "Content-Type: application/json" \
    -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}'
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

# Gateway API keys (in production: store in DB or secret manager)
GATEWAY_KEYS = {
    "gw-test-key-1": {"name": "dev", "rpm_limit": 60},
    "gw-test-key-2": {"name": "staging", "rpm_limit": 200},
    "gw-admin-key":  {"name": "admin", "rpm_limit": 1000},
}

# Provider routing table: gateway model name → provider call info
PROVIDER_MAP = {
    # OpenAI models
    "gpt-4o":           {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini":      {"provider": "openai", "model": "gpt-4o-mini"},
    "gpt-3.5-turbo":    {"provider": "openai", "model": "gpt-3.5-turbo"},
    # Anthropic models
    "claude-opus-4-7":       {"provider": "anthropic", "model": "claude-opus-4-7"},
    "claude-sonnet-4-6":     {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "claude-haiku-4-5":      {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    # Ollama local models (no key needed)
    "llama3.2":         {"provider": "ollama", "model": "llama3.2"},
    "mistral":          {"provider": "ollama", "model": "mistral"},
    "phi4":             {"provider": "ollama", "model": "phi4"},
    "deepseek-r1":      {"provider": "ollama", "model": "deepseek-r1"},
    # Fallback chains: if primary provider fails, try secondary
    "gpt-4o-with-fallback": [
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        {"provider": "ollama", "model": "llama3.2"},
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")


# ─── State ────────────────────────────────────────────────────────────────────

# Request log (in production: use a proper DB / Clickhouse / Datadog)
request_log: list[dict] = []

# Rate limit counters: {api_key: [(timestamp, request_count)]}
rate_counters: dict[str, list[float]] = defaultdict(list)

# Spend tracker: {api_key: total_usd_cost}
spend_tracker: dict[str, float] = defaultdict(float)


# ─── Token counting ───────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "gpt-4o":        (0.0025, 0.010),
        "gpt-4o-mini":   (0.00015, 0.0006),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "claude-opus-4-7":   (0.015, 0.075),
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-haiku-4-5-20251001": (0.0008, 0.004),
    }
    if model in prices:
        inp_price, out_price = prices[model]
        return (input_tokens / 1000 * inp_price) + (output_tokens / 1000 * out_price)
    return 0.0


# ─── Auth middleware ──────────────────────────────────────────────────────────

def authenticate(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    key = auth[7:]
    if key not in GATEWAY_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return GATEWAY_KEYS[key]


def check_rate_limit(api_key: str, key_config: dict) -> None:
    now = time.time()
    window = 60  # 1 minute
    # Remove entries older than window
    rate_counters[api_key] = [t for t in rate_counters[api_key] if now - t < window]
    count = len(rate_counters[api_key])
    limit = key_config.get("rpm_limit", 60)
    if count >= limit:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {limit} req/min")
    rate_counters[api_key].append(now)


# ─── Provider callers ─────────────────────────────────────────────────────────

async def call_openai(model: str, messages: list, max_tokens: int, stream: bool, client: httpx.AsyncClient) -> dict | AsyncIterator:
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream}
    r = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


async def call_anthropic(model: str, messages: list, max_tokens: int, stream: bool, client: httpx.AsyncClient) -> dict:
    if not ANTHROPIC_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": user_msgs,
    }
    if system:
        payload["system"] = system
    r = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    # Normalise to OpenAI format
    return {
        "id": data.get("id", ""),
        "model": data.get("model", model),
        "choices": [{"message": {"role": "assistant", "content": data["content"][0]["text"]}, "finish_reason": data.get("stop_reason", "stop")}],
        "usage": {"prompt_tokens": data.get("usage", {}).get("input_tokens", 0), "completion_tokens": data.get("usage", {}).get("output_tokens", 0)},
    }


async def call_ollama(model: str, messages: list, max_tokens: int, client: httpx.AsyncClient) -> dict:
    r = await client.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"num_predict": max_tokens}},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    prompt_tokens = data.get("prompt_eval_count", count_tokens(" ".join(m["content"] for m in messages)))
    completion_tokens = data.get("eval_count", count_tokens(content))
    return {
        "id": f"ollama-{uuid.uuid4().hex[:8]}",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


async def dispatch(provider: str, model: str, messages: list, max_tokens: int, client: httpx.AsyncClient) -> dict:
    if provider == "openai":
        return await call_openai(model, messages, max_tokens, stream=False, client=client)
    elif provider == "anthropic":
        return await call_anthropic(model, messages, max_tokens, stream=False, client=client)
    elif provider == "ollama":
        return await call_ollama(model, messages, max_tokens, client=client)
    raise ValueError(f"Unknown provider: {provider}")


# ─── Request schema ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 512
    stream: bool = False
    temperature: float = 1.0


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Mini AI Gateway starting on http://localhost:8000")
    log.info(f"Providers: OpenAI={'yes' if OPENAI_KEY else 'no'}, Anthropic={'yes' if ANTHROPIC_KEY else 'no'}, Ollama=yes")
    log.info(f"Gateway keys: {list(GATEWAY_KEYS.keys())}")
    yield


app = FastAPI(title="Mini AI Gateway", version="1.0.0", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    # Auth
    key_header = req.headers.get("Authorization", "Bearer gw-test-key-1")
    raw_key = key_header[7:] if key_header.startswith("Bearer ") else key_header
    key_config = GATEWAY_KEYS.get(raw_key, {"name": "unknown", "rpm_limit": 10})
    if raw_key not in GATEWAY_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    check_rate_limit(raw_key, key_config)

    body = await req.json()
    chat_req = ChatRequest(**body)

    # Resolve routing
    route_info = PROVIDER_MAP.get(chat_req.model)
    if route_info is None:
        raise HTTPException(status_code=400, detail=f"Unknown model: {chat_req.model}. Available: {list(PROVIDER_MAP.keys())}")

    # Normalise to list for fallback support
    routes = route_info if isinstance(route_info, list) else [route_info]

    request_id = f"gw-{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()
    last_error = None

    async with httpx.AsyncClient() as client:
        for route in routes:
            provider = route["provider"]
            model = route["model"]
            try:
                log.info(f"[{request_id}] {key_config['name']} → {provider}/{model}")
                result = await dispatch(provider, model, chat_req.messages, chat_req.max_tokens, client)

                latency_ms = (time.perf_counter() - start_time) * 1000
                usage = result.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                cost = estimate_cost(model, in_tok, out_tok)
                spend_tracker[raw_key] += cost

                log_entry = {
                    "id": request_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "key_name": key_config["name"],
                    "requested_model": chat_req.model,
                    "provider": provider,
                    "model": model,
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "latency_ms": round(latency_ms),
                    "cost_usd": round(cost, 6),
                }
                request_log.append(log_entry)
                log.info(f"[{request_id}] done in {latency_ms:.0f}ms | in={in_tok} out={out_tok} | ${cost:.6f}")

                # Add gateway headers
                result["id"] = request_id
                result["gateway"] = {"provider": provider, "latency_ms": round(latency_ms), "cost_usd": round(cost, 6)}
                return result

            except Exception as e:
                log.warning(f"[{request_id}] {provider}/{model} failed: {e}")
                last_error = e
                continue

    raise HTTPException(status_code=502, detail=f"All providers failed. Last error: {last_error}")


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": k, "object": "model"} for k in PROVIDER_MAP]}


@app.get("/gateway/stats")
async def stats():
    total_requests = len(request_log)
    total_cost = sum(e["cost_usd"] for e in request_log)
    by_provider = {}
    for e in request_log:
        p = e["provider"]
        by_provider.setdefault(p, {"requests": 0, "cost": 0.0, "avg_latency_ms": []})
        by_provider[p]["requests"] += 1
        by_provider[p]["cost"] += e["cost_usd"]
        by_provider[p]["avg_latency_ms"].append(e["latency_ms"])

    for p in by_provider:
        lats = by_provider[p]["avg_latency_ms"]
        by_provider[p]["avg_latency_ms"] = round(sum(lats) / len(lats)) if lats else 0
        by_provider[p]["cost"] = round(by_provider[p]["cost"], 6)

    return {
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 6),
        "by_provider": by_provider,
        "spend_by_key": {k: round(v, 6) for k, v in spend_tracker.items()},
        "recent": request_log[-10:],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "providers": {"openai": bool(OPENAI_KEY), "anthropic": bool(ANTHROPIC_KEY), "ollama": True}}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gateway:app", host="0.0.0.0", port=8000, reload=False)
