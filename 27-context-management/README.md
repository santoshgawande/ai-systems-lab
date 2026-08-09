# Section 27 — Context Management

Keep long conversations within token limits without losing important information.

## What you learn

- Sliding window: discard oldest messages when approaching the limit
- Summarisation: compress old turns into a running summary
- Human-in-the-loop: pause and ask when the agent is uncertain
- Token counting across providers

## Labs

| Lab | What it covers |
|---|---|
| 01-sliding-window | Token counting, window truncation, oldest-first eviction |
| 02-summarization | Running summary of old turns, inject summary as context |
| 03-human-in-the-loop | Uncertainty detection, clarification requests, approval gates |

## Setup

```bash
pip install -r requirements.txt
```

## Context window reference (2025)

| Model | Context | Cost (input per 1M tokens) |
|-------|---------|---------------------------|
| GPT-4o | 128K | $2.50 |
| Claude Opus 4.7 | 200K | $15.00 |
| Gemini 2.0 Flash | 1M | $0.10 |
| Llama 3.2 (Ollama) | 128K | Free |

## Why context management matters

A 1-hour support chat might accumulate 50K tokens. Sending the full history on every turn:
- Costs more with each message
- Slows responses as context grows
- Eventually hits the model's limit

The fix is always one of: truncate, summarise, or retrieve.
