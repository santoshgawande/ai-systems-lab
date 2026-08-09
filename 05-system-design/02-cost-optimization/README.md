# Lab 02 — Cost Optimization

Token budgeting, model routing, and prompt caching. Reduce per-call cost by 60-80% with these techniques.

## What you learn

- How to estimate token count and cost before making a call
- How to route cheap vs expensive tasks to the right model tier
- How an in-memory cache eliminates cost for repeated prompts
- The real cost difference between phi4 vs llama3.3:70b

## Run

```bash
python cost.py
```

## Routing rules of thumb

| Task type | Model | Why |
|---|---|---|
| Classification, yes/no, format | phi4 (cheap) | Simple pattern matching |
| Summarization, extraction | llama3.2 (balanced) | Moderate reasoning needed |
| Analysis, architecture, debugging | llama3.3:70b (premium) | Complex reasoning required |

## Token cost model

```
cost = (input_tokens / 1000) * cost_per_1k_input
     + (output_tokens / 1000) * cost_per_1k_output
```

Output is always more expensive than input — minimize output with tight instructions.
