# Lab 01 — Sliding Window

Evict oldest messages when approaching the token limit. The simplest context strategy.

## What you learn

- Token counting: `tiktoken` exact vs char/4 approximation
- Sliding window: evict oldest non-system messages first
- When to use sliding window vs summarisation

## Run

```bash
pip install httpx tiktoken   # tiktoken optional but more accurate
python sliding_window.py
```

## Core functions

```python
def count_tokens(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

def apply_sliding_window(messages, max_tokens):
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]

    while sum(tokens(m) for m in system + history) > max_tokens:
        history.pop(0)   # evict oldest

    return system + history
```

## When to use sliding window

| Scenario | Use sliding window? |
|----------|-------------------|
| Simple Q&A chatbot | Yes |
| Each turn is independent | Yes |
| Conversation builds on history | No — use summarisation |
| Agent with long tool results | No — compress tool outputs |

## Rule of thumb

Reserve `max_tokens = context_limit - output_reserve` for history.
- 128K context, 2K output reserve → 126K for history
- At ~500 tokens/turn, that's 252 turns before eviction starts
