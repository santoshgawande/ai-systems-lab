# Lab 01 — Claude API Basics

The Anthropic Messages API: how it differs from OpenAI, system prompt placement, tool use, and vision.

## What you learn

- The `system` parameter vs OpenAI's `messages[0].role = "system"`
- `input_tokens` / `output_tokens` vs OpenAI's `prompt_tokens` / `completion_tokens`
- How Claude's tool use format compares to OpenAI function calling
- How to send an image to Claude (base64 or URL)

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python claude_basics.py
```

Falls back to Ollama format explanation if no API key set.

## API shape comparison

```python
# OpenAI
client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"}
    ]
)

# Anthropic — system is a SEPARATE parameter, not in messages
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are helpful.",          # ← separate param
    messages=[{"role": "user", "content": "Hi"}]
)
```
