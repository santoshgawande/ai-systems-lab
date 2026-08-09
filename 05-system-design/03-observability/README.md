# Lab 03 — Observability

Instrument every LLM call with structured logging, latency tracking, and token metrics.

## What you learn

- How to capture time-to-first-token (TTFT) vs total latency separately
- How to compute p50/p95/p99 latency across a batch of calls
- Why structured JSONL logging beats print statements for production debugging
- What fields every LLM trace should include

## Run

```bash
python trace.py

# Inspect the raw trace log
cat /tmp/llm_traces.jsonl | python -m json.tool
```

## What to log on every LLM call

```json
{
  "trace_id": "a1b2c3d4",
  "model": "phi4",
  "input_tokens": 45,
  "output_tokens": 12,
  "latency_ms": 843,
  "ttft_ms": 210,
  "status": "ok",
  "error": null,
  "timestamp": "2025-05-07T10:23:14"
}
```

TTFT is the UX metric for streaming. Total latency is the throughput metric.
Track both separately — a slow model with fast first-token feels faster to users.
