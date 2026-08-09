# Lab 01 — Few-Shot Prompting

Adding examples to the context is the single most reliable way to enforce output format.

## What you learn

- Why zero-shot produces inconsistent formats that break parsers
- How 3-4 labeled examples dramatically improve consistency
- Why examples outperform long instruction paragraphs
- How to pick examples that cover edge cases

## Run

```bash
python few_shot.py
```

## Key insight

The model is a pattern-completion engine. Give it a pattern to complete.

Zero-shot: "Classify as POSITIVE, NEGATIVE, or NEUTRAL."
→ Model might say "This is positive!" or "Sentiment: positive." or "Positive."

Few-shot with 3 examples:
→ Model says "POSITIVE." — every time.
