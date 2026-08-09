# Section 13 — Fine-Tuning

Teach a model new behavior through training, not just prompting.

## What you learn

- When fine-tuning beats prompt engineering (and when it doesn't)
- How to create custom Ollama models with Modelfiles
- LoRA / QLoRA — parameter-efficient fine-tuning for open-source models
- Dataset preparation — JSONL format, quality filtering, deduplication

## Labs

| Lab | What it covers |
|---|---|
| 01-when-to-fine-tune | Decision framework: fine-tune vs prompt vs RAG |
| 02-ollama-modelfiles | Build custom models with Ollama Modelfiles |
| 03-lora-concepts | LoRA theory, PEFT, QLoRA — adapt large models cheaply |

## Setup

```bash
pip install -r requirements.txt
# Requires Ollama running at localhost:11434
```

## Fine-tuning decision tree

```
New task ──► Prompt engineering works?
               │ YES → Done. Use prompting.
               │ NO
               ▼
          Do you have 50–1000 examples?
               │ NO  → Collect data first
               │ YES
               ▼
          Need consistent style/format?  → Fine-tuning
          Need new knowledge?           → Fine-tuning
          Need flexible reasoning?      → RAG + Prompting
          Need fast iteration?          → Prompting
```

## Cost comparison (GPT-4o-mini fine-tuning)

| Approach | Cost per 1M tokens | Latency |
|---|---|---|
| GPT-4o-mini base | $0.15 in / $0.60 out | ~200ms |
| GPT-4o-mini fine-tuned | $0.30 in / $1.20 out | ~200ms |
| GPT-4o | $2.50 in / $10.00 out | ~400ms |

Fine-tuning lets a cheap model behave like an expensive one for your specific task.
