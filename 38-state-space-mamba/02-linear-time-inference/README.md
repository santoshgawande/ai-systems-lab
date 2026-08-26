# Lab 02: Mamba Constant-Memory Inference

## What You Learn
- Why autoregressive generation in Transformers suffers from linear $O(N)$ memory growth.
- How Mamba generates tokens with strictly constant $O(1)$ GPU state memory.
- Latency and memory benchmarks across extended generation lengths.

## Run
```bash
python 02-linear-time-inference/mamba_inference.py
```
