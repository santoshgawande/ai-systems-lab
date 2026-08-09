# Section 22 — LiteLLM

One interface for every LLM provider. Switch models without rewriting your code.

## What you learn

- LiteLLM unified API: same call works for OpenAI, Anthropic, Gemini, Ollama
- Cost tracking, spend limits, logging built-in
- Router: load balancing, fallbacks, retries across providers

## Labs

| Lab | What it covers |
|---|---|
| 01-basics | Unified API, provider switching, token cost tracking |
| 02-routing | Router with fallbacks, rate-limit handling, load balancing |

## Setup

```bash
pip install -r requirements.txt
```

## Why LiteLLM

Without LiteLLM, switching providers requires rewriting API calls:
```python
# OpenAI
client.chat.completions.create(model="gpt-4o", ...)

# Anthropic  
client.messages.create(model="claude-opus-4-7", ...)

# Gemini
model.generate_content(...)
```

With LiteLLM, every provider uses the same call:
```python
litellm.completion(model="gpt-4o", messages=[...])
litellm.completion(model="claude-opus-4-7", messages=[...])
litellm.completion(model="gemini/gemini-2.0-flash", messages=[...])
litellm.completion(model="ollama/llama3.2", messages=[...])
```
