# Lab 01 — OpenAI Batch API

Process thousands of LLM requests at 50% lower cost with a 24-hour turnaround.

## What you learn

- Batch file format: JSONL with `custom_id`, `method`, `url`, `body`
- Creating a batch job, polling status, retrieving results
- Matching results back to input via `custom_id`
- Error handling: partial failures within a batch

## Run

```bash
export OPENAI_API_KEY=sk-...
python batch.py
```

## When batch beats real-time

| | Real-time API | Batch API |
|---|---|---|
| Latency | ~300ms–2s | Up to 24h |
| Cost | Standard | 50% off |
| Rate limits | Tight | Relaxed (up to 100K req/batch) |
| Best for | Chat, interactive | Nightly ETL, eval suites, bulk classification |

## Input format (JSONL)

Each line is one request:
```json
{"custom_id": "doc-001", "method": "POST", "url": "/v1/chat/completions", "body": {
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Classify: 'I love this product!' → positive/negative/neutral"}],
  "max_tokens": 10
}}
```

## Output format

```json
{"id": "batch_req_abc", "custom_id": "doc-001", "response": {
  "status_code": 200,
  "body": {"choices": [{"message": {"content": "positive"}}]}
}}
```

## Polling states

```
validating → in_progress → completed
                        └→ failed (entire batch)
```
