# Mini AI Gateway

A production-pattern LLM proxy in ~300 lines. Routes requests across OpenAI, Anthropic, and Ollama with auth, logging, rate limiting, and automatic fallback.

## What it teaches

- How LiteLLM, Portkey, and OpenRouter work under the hood
- API key authentication and per-key rate limiting
- Provider routing table: logical model name → actual provider call
- Normalising different provider response formats to one schema
- Automatic fallback chains: if OpenAI fails → try Anthropic → try Ollama
- Request logging with token counts and cost estimation
- FastAPI middleware patterns for cross-cutting concerns

## Run

```bash
pip install fastapi uvicorn httpx pydantic tiktoken
export OPENAI_API_KEY=sk-...       # optional
export ANTHROPIC_API_KEY=sk-...    # optional
python gateway.py
# Server starts on http://localhost:8000
```

## Use it

```bash
# Call through the gateway (same shape as OpenAI API)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer gw-test-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is a transformer?"}],
    "max_tokens": 100
  }'

# Route to Ollama (no API key needed)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer gw-test-key-1" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]}'

# List available models
curl http://localhost:8000/v1/models

# Gateway stats (requests, cost, latency per provider)
curl http://localhost:8000/gateway/stats

# Health check
curl http://localhost:8000/health
```

## What's built in

| Feature | Implementation |
|---------|---------------|
| Auth | `Authorization: Bearer gw-key` header validation |
| Rate limiting | Sliding window counter per API key (req/min) |
| Routing table | `PROVIDER_MAP` dict: logical model → provider+model |
| Fallback chain | List of routes: tries each until one succeeds |
| Response normalisation | Anthropic/Ollama responses converted to OpenAI shape |
| Token counting | tiktoken for OpenAI; native counts from Ollama |
| Cost estimation | Per-model price table, tracked per key |
| Request log | In-memory list with all request metadata |
| Stats endpoint | `/gateway/stats` for cost/latency breakdown |

## Architecture

```
Client
  ↓  Authorization: Bearer gw-test-key-1
FastAPI /v1/chat/completions
  ↓  authenticate() + check_rate_limit()
  ↓  resolve model → provider(s)
  ↓  try provider 1 → fail? try provider 2 → ...
OpenAI / Anthropic / Ollama
  ↓  normalise response to OpenAI format
  ↓  log request (tokens, cost, latency)
Client ← response + gateway metadata headers
```

## Extend it

- **Persistent logging**: swap `request_log` list for PostgreSQL/ClickHouse
- **Semantic caching**: hash prompt → check Redis before calling provider
- **Spend limits**: block keys that exceed monthly budget
- **Streaming**: add `stream: true` support with SSE passthrough
- **RBAC**: different keys get different model access
- **Admin UI**: add a `/dashboard` HTML endpoint over the stats data
