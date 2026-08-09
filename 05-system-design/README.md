# 05 — System Design for AI

How production AI systems handle reliability, cost, observability, and routing at scale.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-reliability/` | Exponential backoff with jitter, fallback routing, circuit breaker | `python resilient_client.py` |
| `02-cost-optimization/` | Token budgeting, model routing by task complexity, prompt caching | `python cost.py` |
| `03-observability/` | Structured request logging, TTFT, p50/p95/p99 latency tracking | `python trace.py` |
| `04-gateway/` | LLM proxy: API key auth, model routing, rate limiting, logging | `python gateway.py` |

## Key concepts

- Rate limit errors (429) require exponential backoff with **jitter** — never plain `sleep(retry * 2)`
- Model routing: use cheap (phi4) for classification/extraction, expensive (llama3.3:70b) for reasoning
- Time-to-first-token (TTFT) is the perceived UX latency for streaming — track it separately from total latency
- Every LLM call should log: model, input_tokens, output_tokens, latency_ms, ttft_ms, status, error
- A gateway centralizes auth + routing + rate limiting so every service doesn't re-implement them
