# Lab 01: Sparse MoE Top-K Gating Router

## What You Learn
- How Sparse MoE scales total parameters without scaling FLOPs per token.
- Top-K expert selection ($K=2$ or $K=8$) and softmax weight normalization.
- DeepSeek-V3 shared expert isolation for foundational universal knowledge.
- Calculating active parameter efficiency.

## Run
```bash
python 01-topk-gating/moe_gating.py
```
