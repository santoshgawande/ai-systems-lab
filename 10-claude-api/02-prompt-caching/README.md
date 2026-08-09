# Lab 02 — Prompt Caching

Claude's prompt caching reduces cost by 90% for repeated large system prompts — the feature that makes Claude Code economically viable.

## What you learn

- How to mark prompt sections with `cache_control: {type: "ephemeral"}`
- How to measure `cache_creation_input_tokens` vs `cache_read_input_tokens`
- Why this matters for Claude Code (50k+ token system prompt on every request)
- The 5-minute cache TTL and what it means for your usage pattern

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python caching.py
```

## How it works

```python
# Mark a large system prompt section for caching
system=[
    {
        "type": "text",
        "text": LARGE_KNOWLEDGE_BASE,          # 10k+ tokens
        "cache_control": {"type": "ephemeral"} # ← cache this prefix
    },
    {
        "type": "text",
        "text": "Answer from the above context only.",
    }
]
```

First request: `cache_creation_input_tokens` = full tokens (priced at 1.25x)
Second+ request within 5 min: `cache_read_input_tokens` = full tokens (priced at 0.10x)

## Cost math

| Request | Input tokens | Cost (Sonnet) |
|---|---|---|
| No cache, 50k system prompt | 50,000 | $0.150 |
| Cache write (1.25x) | 50,000 | $0.1875 |
| Cache read (0.10x) | 50,000 | $0.015 |

After the first call, every repeat = **90% cheaper**.
