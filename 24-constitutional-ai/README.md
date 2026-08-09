# Section 24 — Constitutional AI & Moderation

Make LLM outputs self-correct against a set of principles, and use moderation APIs to filter harmful content.

## What you learn

- Constitutional AI: critique → revise loop using principles
- Self-critique pattern: ask the model to find its own problems, then fix them
- OpenAI Moderation API: classify harm categories with confidence scores
- When to use each approach in production

## Labs

| Lab | What it covers |
|---|---|
| 01-self-critique | CAI principles, critique-revise loop, before/after comparison |
| 02-moderation-api | OpenAI moderation endpoint, harm categories, flagging pipeline |

## Setup

```bash
pip install -r requirements.txt
```

## Constitutional AI overview

Original paper (Anthropic, 2022): train models on a set of principles by having them:
1. **Critique** their own output against each principle
2. **Revise** the output to fix the critique
3. Repeat until no violations found

In production, you don't need to retrain — just run this loop at inference time.
