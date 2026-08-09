# Lab 03 — Async Queue Patterns

When you have high-volume AI workloads, you need a queue — not a synchronous API call.

## What you learn

- Why direct API calls fail at high volume (rate limits, timeouts, cost spikes)
- asyncio + semaphore for controlled concurrent LLM calls
- Queue-worker pattern: producer enqueues, workers process, results collected
- Priority queuing: urgent requests jump the queue

## Run

```bash
python async_queue.py
# Uses Ollama — no API key needed
```

## Pattern comparison

| Pattern | When to use |
|---|---|
| Synchronous call | Single request, interactive |
| Batch API | Offline processing, 24h SLA OK, want 50% savings |
| Async queue | Real-time processing at scale, mixed priority, need results ASAP |

## Async queue architecture

```
Producers                Queue              Workers              Results
─────────────            ─────────          ─────────            ───────
job_request_1 ──────►   [P3, P1, P2]  ──►  worker_1  ──────►  result_1
job_request_2 ──────►                  ──►  worker_2  ──────►  result_2
job_request_3 ──────►                  ──►  worker_3  ──────►  result_3
```

## Concurrency control

```python
# Token bucket: N concurrent LLM calls max
semaphore = asyncio.Semaphore(5)  # max 5 parallel calls

async def call_with_limit(prompt):
    async with semaphore:           # blocks if 5 already running
        return await llm.call(prompt)
```

## Rate limit recovery

```python
async def call_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await llm.call(prompt)
        except RateLimitError:
            wait = (2 ** attempt) + random.uniform(0, 1)  # exp backoff + jitter
            await asyncio.sleep(wait)
    raise MaxRetriesExceeded()
```
