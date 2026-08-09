# Lab 04 — LLM API Gateway

A minimal FastAPI gateway that adds auth, model routing, rate limiting, and logging in front of Ollama.

## What you learn

- How LiteLLM, Portkey, and OpenRouter work internally
- How to map OpenAI-style model names (gpt-4o) to local models (llama3.3:70b)
- How per-user rate limiting works with a token bucket
- Why centralizing this in a gateway beats doing it in every service

## Run

```bash
pip install fastapi uvicorn httpx
python gateway.py
```

## Test it

```bash
# Health check
curl http://localhost:8080/health

# Chat (OpenAI-compatible format)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-lab-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'

# Streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-lab-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

## API keys

| Key | User |
|---|---|
| `sk-lab-key-1` | user1 |
| `sk-lab-key-2` | user2 |

## Model routing

| Requested | Routes to |
|---|---|
| gpt-4, gpt-4o, claude-3-5 | llama3.3:70b |
| gpt-4o-mini, claude-haiku | phi4 |
| anything else | llama3.2 |
