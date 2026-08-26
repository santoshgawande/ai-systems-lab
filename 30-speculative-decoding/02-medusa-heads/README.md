# Lab 02: Medusa Multi-Head Decoding

## What You Learn
- How Medusa uses multiple prediction heads on a single base model rather than maintaining a separate draft model.
- Candidate tree generation across heads: $t+1, t+2, t+3$.
- Tree attention verification and longest valid prefix acceptance.
- How to eliminate draft model memory overhead and synchronization latency.

## Run
```bash
python 02-medusa-heads/medusa.py
```
