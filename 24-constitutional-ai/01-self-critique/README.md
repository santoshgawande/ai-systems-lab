# Lab 01 — Constitutional AI Self-Critique

Teach an LLM to critique and revise its own outputs against a set of principles.

## What you learn

- The CAI loop: initial response → critique → revise → repeat
- Writing a "constitution" of principles
- When to use CAI vs a fine-tuned classifier
- Production tradeoffs: latency, cost, when to stop iterating

## Run

```bash
pip install httpx
python self_critique.py
# Works with Ollama (default), OpenAI, or Anthropic
```

## The CAI loop

```python
response = llm(user_request)                    # initial response

for principle in principles:
    critique = llm(f"""
        Does this violate: "{principle}"?
        Response: {response}
        Say "No violation." or explain the issue.
    """)

    if "no violation" not in critique.lower():
        response = llm(f"""
            Rewrite to comply with: "{principle}"
            Violation: {critique}
            Original: {response}
        """)
```

## Constitutional principles examples

```python
PRINCIPLES = [
    "Do not provide instructions that could harm someone.",
    "Do not present speculation as fact without qualification.",
    "Do not demean people based on identity.",
    "Respect privacy — no surveillance suggestions without consent.",
]
```

## When to use CAI

| Scenario | Use CAI? |
|----------|----------|
| Low-volume, high-stakes (legal, medical) | Yes |
| Catching model drift in production | Yes |
| High-volume API (>1000 req/s) | No — too slow |
| Simple offensive content filtering | No — use moderation API |
| Custom brand voice enforcement | Yes |

## Production optimisations

- Run critiques **in parallel** with `asyncio.gather()`
- Only revise if a violation is found (saves 1 LLM call per clean principle)
- Keep the constitution to ≤5 principles to stay under 3s latency
