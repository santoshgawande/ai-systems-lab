# Lab 05 — Benchmark

Measure latency and throughput for each local model. Run this before picking a model for a production feature.

## What you learn

- p50 / p95 latency — why average is misleading
- Tokens per second — the throughput metric for LLMs
- Time-to-first-token (TTFT) vs total duration
- How model size, quantisation, and hardware affect performance

## Run

```bash
pip install requests rich
python benchmark.py                                      # default models + prompt
python benchmark.py --models phi4 llama3.2 --trials 3
python benchmark.py --prompt "Summarise the history of the internet in 5 bullet points"
```

## Metrics explained

| Metric | What it means |
|--------|--------------|
| p50 latency | Median response time — "typical" user experience |
| p95 latency | 95th percentile — worst case 1 in 20 requests |
| tokens/sec | Output throughput — larger = faster typing experience |
| TTFT | Time until first token arrives — perceived responsiveness |

## Reading results

```
Model             p50     p95    tok/s
phi4              0.8s    1.1s   45
llama3.2          1.2s    1.8s   38
llama3.3:70b      4.1s    5.3s   18
```

A fast small model for a latency-sensitive feature, large model for accuracy-sensitive batch work.

## When to benchmark

- Before adding a model to production
- After hardware changes (new GPU, more RAM)
- When comparing quantisation levels (Q4 vs Q8 vs FP16)
