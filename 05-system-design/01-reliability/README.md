# Lab 01 — Reliability: Retry, Backoff, and Fallback

Production LLM clients need to handle transient failures gracefully. See how to implement it right.

## What you learn

- Why plain `sleep(retry_count * 2)` causes thundering herd — use **jitter**
- The difference between retryable (429, 503, timeout) and non-retryable (400, 401) errors
- How fallback routing automatically tries the next model when the primary fails
- What a circuit breaker looks like for LLM calls

## Run

```bash
python resilient_client.py
```

## Exponential backoff with jitter

```python
delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
```

Without jitter: all retries happen at the same time → thundering herd → 429 cascade.
With jitter: retries spread out → pressure drops → requests eventually succeed.

## When to retry vs fail fast

| Error | Retry? | Why |
|---|---|---|
| 429 (rate limit) | Yes | Temporary, will clear |
| 503 (server error) | Yes | Temporary overload |
| Timeout | Yes | Transient network issue |
| 400 (bad request) | No | Your request is wrong, retrying won't help |
| 401 (unauthorized) | No | Auth problem, not transient |
