# Lab 01 — LiteLLM Basics

One API call that works for OpenAI, Anthropic, Gemini, and Ollama.

## What you learn

- `litellm.completion()` as a universal drop-in
- Automatic cost tracking for cloud models
- Streaming with `stream=True` (same flag for all providers)
- Async with `acompletion()` for concurrent calls

## Run

```bash
pip install litellm
export OPENAI_API_KEY=sk-...       # optional
export ANTHROPIC_API_KEY=sk-...    # optional
python litellm_basics.py
# Ollama demo runs without any key
```

## Key API

```python
import litellm

# Sync
resp = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
print(resp.choices[0].message.content)

# Cost (automatic for cloud providers)
print(litellm.completion_cost(resp))  # "$0.000030"

# Stream
for chunk in litellm.completion(model="gpt-4o-mini", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# Async
resp = await litellm.acompletion(model="claude-haiku-4-5-20251001", messages=[...])
```

## Provider strings

| Provider | Model string |
|----------|-------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | `claude-opus-4-7`, `claude-haiku-4-5-20251001` |
| Gemini | `gemini/gemini-2.0-flash` |
| Ollama | `ollama/llama3.2` (+ `api_base=`) |
| Azure | `azure/<deployment-name>` |
