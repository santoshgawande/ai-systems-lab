# Lab 01 — When to Fine-Tune

The most expensive mistake in AI: fine-tuning when a better prompt would have worked.

## What you learn

- The decision framework: prompt engineering → RAG → fine-tuning
- What fine-tuning is actually doing (gradient updates on adapter weights)
- Dataset requirements: how much data is "enough"
- Cost model: training job + inference premium vs prompting

## Run

```bash
python decision.py
```

## Decision framework

```
Task defined?
  └─ Can a good prompt do it?       → Use prompting (iterate fast)
      └─ Need private knowledge?    → Add RAG (no training needed)
          └─ Need consistent style? → Fine-tune (bake format into weights)
          └─ Dataset < 50 examples? → Collect more data first
          └─ Need new knowledge?    → Fine-tune OR RAG (RAG is cheaper)
```

## Signals that suggest fine-tuning

- Prompt is > 2000 tokens to establish behavior (expensive, fragile)
- You need EXACT output format every time (no variation)
- Task requires domain-specific vocabulary/reasoning not in base model
- Cost reduction: cheap model + fine-tuning beats expensive base model

## Signals that suggest prompting/RAG instead

- Behavior changes weekly (fine-tuning is slow to update)
- Need to reference specific documents/facts
- Limited training data (< 50 high-quality examples)
- Iterating on the task definition
