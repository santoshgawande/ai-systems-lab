# Section 15 — AI Security

The OWASP Top 10 for LLMs, plus prompt injection and red-teaming techniques.

## What you learn

- OWASP LLM Top 10 — the 10 most critical LLM application vulnerabilities
- Prompt injection — direct vs indirect, detection patterns, mitigations
- Red teaming — systematic adversarial testing of your AI application

## Labs

| Lab | What it covers |
|---|---|
| 01-owasp-llm-top10 | All 10 risks with code examples and mitigations |
| 02-prompt-injection | Direct + indirect injection attacks and defenses |
| 03-red-teaming | Systematic adversarial testing framework |

## Setup

```bash
pip install -r requirements.txt
# Uses Ollama at localhost:11434 — no API key needed
```

## OWASP LLM Top 10 (2025)

| # | Risk | Short description |
|---|---|---|
| LLM01 | Prompt Injection | Malicious input hijacks model behavior |
| LLM02 | Insecure Output Handling | Model output used unsanitized (XSS, SQLi) |
| LLM03 | Training Data Poisoning | Corrupted training data manipulates behavior |
| LLM04 | Model Denial of Service | Expensive queries exhaust resources |
| LLM05 | Supply Chain Vulnerabilities | Compromised models, datasets, plugins |
| LLM06 | Sensitive Information Disclosure | Model reveals training data / system prompts |
| LLM07 | Insecure Plugin Design | Plugins with excessive permissions |
| LLM08 | Excessive Agency | Agent takes unintended destructive actions |
| LLM09 | Overreliance | Users trust hallucinated output |
| LLM10 | Model Theft | Extraction of model weights or behavior |

## Difference from 09-guardrails

- 09-guardrails: code-level input/output validation
- 15-ai-security: attack taxonomy, adversarial testing, systematic hardening
