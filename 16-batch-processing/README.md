# Section 16 — Batch Processing

Process thousands of LLM requests at 50% lower cost by using async batch APIs.

## What you learn

- OpenAI Batch API — submit JSONL, poll for completion, 50% cost reduction
- Anthropic Message Batches — same pattern, different API shape
- Async queue patterns — when to batch vs stream vs queue

## Labs

| Lab | What it covers |
|---|---|
| 01-openai-batch | OpenAI Batch API: create batch, poll status, retrieve results |
| 02-anthropic-batch | Anthropic Message Batches API: same workflow, Anthropic SDK |
| 03-async-queue | Queue-based async AI processing for high-volume workloads |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## When to use batch processing

| Use case | Real-time API | Batch API |
|---|---|---|
| Interactive chat | ✓ | — |
| Document classification (1000s) | Too expensive | ✓ |
| Nightly data enrichment | — | ✓ |
| Eval suite execution | — | ✓ |
| Latency SLA < 5s | ✓ | — |
| Cost is primary constraint | — | ✓ |

## Cost comparison (OpenAI gpt-4o-mini)

| | Standard API | Batch API |
|---|---|---|
| Input cost | $0.15/1M | $0.075/1M (50% off) |
| Output cost | $0.60/1M | $0.30/1M (50% off) |
| Latency | <1s | Up to 24h |
| Max requests | Rate limited | Up to 50,000/batch |

## Batch file format (JSONL)

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Classify: positive or negative? 'This product is great!'"}]}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Classify: positive or negative? 'Worst purchase ever.'"}]}}
```
