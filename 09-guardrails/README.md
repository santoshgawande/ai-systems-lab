# 09 — Guardrails

Input validation and output safety for production AI systems. Defense in depth: check before you send, and check before you use.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Ollama at `http://localhost:11434`.

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-input-guards/` | Detect prompt injection, PII, jailbreak patterns before hitting the LLM | `python input_guard.py` |
| `02-output-guards/` | Validate JSON schema, detect hallucination signals, filter unsafe output | `python output_guard.py` |

## Why guardrails matter

- **Prompt injection**: users craft input that overrides your system prompt ("ignore previous instructions...")
- **PII leakage**: user sends personal data you didn't intend to forward to an external API
- **Jailbreaks**: carefully crafted prompts bypass safety training
- **Bad JSON**: if your app expects JSON and the model outputs prose, your code crashes
- **Hallucination**: model confidently invents facts — you need to detect and handle this

## Layers

```
User input → [Input Guard] → LLM → [Output Guard] → Application
```

Neither layer is a silver bullet. Layer them.
