# Lab 02 — Streaming

LLMs send tokens one at a time over HTTP. This lab shows how that works at the wire level.

## What you learn

- How streaming works: `stream: true` → server sends newline-delimited JSON chunks
- Each chunk has `message.content` with the next token fragment
- Time-to-first-token (TTFT) vs total latency — why streaming feels faster
- How ChatGPT, Claude, and Copilot all use the same SSE pattern

## Run

```bash
pip install requests
python stream.py
python stream.py "What is RAG?"
```

## Wire format

With `stream: true`, each line is one JSON chunk:

```
{"model":"llama3.2","message":{"role":"assistant","content":"The"},"done":false}
{"model":"llama3.2","message":{"role":"assistant","content":" trans"},"done":false}
{"model":"llama3.2","message":{"role":"assistant","content":"former"},"done":false}
...
{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"eval_count":42}
```

## Read the stream

```python
resp = requests.post(url, json={..., "stream": True}, stream=True)
for line in resp.iter_lines():
    chunk = json.loads(line)
    print(chunk["message"]["content"], end="", flush=True)
    if chunk["done"]:
        break
```

## Key insight

Streaming is essential for UX: a 3-second response feels instant when tokens appear immediately.
TTFT is often more important than total latency for user-facing products.
