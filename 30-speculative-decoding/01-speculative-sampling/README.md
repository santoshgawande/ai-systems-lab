# Lab 01: Speculative Sampling Engine

## What You Learn
- How speculative decoding accelerates LLM generation by $2\times$ to $3\times$ without altering output distribution.
- The mathematical formulation of rejection sampling: $r < \min(1, q(x)/p(x))$.
- Residual distribution resampling upon token rejection: $\max(0, q(x) - p(x))$.
- Empirical speedup calculations and acceptance rate tracking.

## Run
```bash
python 01-speculative-sampling/speculative_sampling.py
```
