# Lab 01: Test-Time Compute (TTC) & Thinking Budget

## What You Learn
- How modern reasoning models (DeepSeek-R1, OpenAI o1/o3) use Test-Time Compute scaling.
- Separating internal reasoning traces (`<think> ... </think>`) from user-facing responses.
- Detecting self-correction, backtracking, and verification triggers in thought logs.
- Dynamically allocating inference token budgets based on query complexity.

## Run
```bash
python 01-test-time-compute/budget_forcing.py
```
