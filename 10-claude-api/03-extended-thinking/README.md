# Lab 03 — Extended Thinking

Claude's "think before answering" mode — allocate token budget for internal reasoning before producing the response.

## What you learn

- How to enable extended thinking with `thinking: {type: "enabled", budget_tokens: N}`
- How thinking tokens appear as a `thinking` content block (not billed the same as output)
- Why thinking improves accuracy on complex reasoning, math, and code tasks
- The trade-off: more thinking = slower + more expensive, but more accurate

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python thinking.py
```

## How it works

```python
response = client.messages.create(
    model="claude-sonnet-4-6",          # or claude-opus-4-7
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000           # how much to think before answering
    },
    messages=[{"role": "user", "content": hard_problem}]
)

for block in response.content:
    if block.type == "thinking":
        print(f"[Claude's reasoning]: {block.thinking}")
    elif block.type == "text":
        print(f"[Answer]: {block.text}")
```

## When to use extended thinking

- Multi-step math or algorithm problems
- Code with complex logic or multiple edge cases
- Comparing options with many trade-offs
- Problems where you need the reasoning visible for audit

## Cost

Budget tokens are billed at input token rates.
A 5000 token thinking budget = ~$0.015 per call with Sonnet.
Only use it when the extra accuracy justifies the cost.
