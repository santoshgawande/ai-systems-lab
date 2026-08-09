# Lab 02 — Chain-of-Thought (CoT)

"Think step by step" is one of the highest-leverage prompt techniques. It trades token cost for accuracy.

## What you learn

- Why direct answers fail on multi-step reasoning problems
- How CoT forces the model to surface its reasoning
- The difference between zero-shot CoT ("think step by step") and few-shot CoT (worked examples)
- When CoT helps vs when it doesn't

## Run

```bash
python cot.py
```

## How it works

Direct: "What is 15% of 847?" → Model might say 127.05 (correct) or 120 (wrong)
CoT: "Think step by step" → Model says:
  1. 10% of 847 = 84.7
  2. 5% = half of that = 42.35
  3. Total = 84.7 + 42.35 = 127.05

The reasoning process catches arithmetic errors before the final answer.

## When CoT helps

- Multi-step arithmetic
- Logical reasoning (if A then B, if B then C...)
- Code debugging
- Comparing options with trade-offs

## When CoT doesn't help

- Simple factual recall ("What is the capital of France?")
- Creative writing
- Format conversion (CSV → JSON)
