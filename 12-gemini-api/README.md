# 12 — Gemini API (Google)

Gemini-specific features: 1M token context window, multimodal inputs (text + image + video), and grounding with Google Search.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get an API key: https://aistudio.google.com/apikey

```bash
export GEMINI_API_KEY=AIza...
```

> Labs fall back to Ollama for conceptual equivalents when key is not set. Long context (>200k tokens) and grounding require a real Gemini key.

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-long-context/` | Process large documents in Gemini's 1M token window; lost-in-the-middle problem | `python long_context.py` |
| `02-multimodal/` | Send text + image to Gemini; cross-modal reasoning | `python multimodal.py` |
| `03-grounding/` | Ground answers in live Google Search results; compare grounded vs ungrounded | `python grounding.py` |

## Gemini model tiers (2025)

| Model | Context | Strengths |
|---|---|---|
| `gemini-2.0-flash` | 1M tokens | Fast, cheap, multimodal |
| `gemini-2.0-flash-thinking` | 32k | Step-by-step reasoning (like o1) |
| `gemini-1.5-pro` | 2M tokens | Largest context window |

## Why Gemini's long context changes the game

Traditional RAG: chunk → embed → retrieve → inject top-K chunks.
Problem: you lose context from chunks that don't rank highly.

With 1M token context: just put the entire document in the prompt.
No chunking, no retrieval, no missed context.
Trade-off: slower, more expensive, and "lost in the middle" still affects quality.
