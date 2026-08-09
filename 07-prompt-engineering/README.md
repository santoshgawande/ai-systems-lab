# 07 — Prompt Engineering

Prompt engineering is structured communication — not magic. These labs show the techniques that turn inconsistent LLM outputs into reliable, parseable, production-grade responses.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Ollama at `http://localhost:11434`.

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-few-shot/` | Why 3 examples beats a paragraph of instructions | `python few_shot.py` |
| `02-chain-of-thought/` | Why "think step by step" fixes reasoning failures | `python cot.py` |
| `03-structured-output/` | Reliable JSON extraction — schema, extraction, parse recovery | `python structured.py` |
| `04-system-prompt-design/` | Weak vs strong system prompts — role, constraints, format, examples | `python system_prompt.py` |

## Key concepts

- **Zero-shot**: just an instruction. Works for simple tasks, inconsistent on edge cases.
- **Few-shot**: instruction + examples. Dramatically improves format consistency.
- **Chain-of-thought**: ask the model to reason before answering. Fixes multi-step logic failures.
- **Structured output**: tight schema + extraction fallback = parseable output you can rely on in production.
- System prompt = contract. The more precise the contract, the more consistent the behavior.
