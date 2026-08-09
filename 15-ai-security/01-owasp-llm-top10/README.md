# Lab 01 — OWASP LLM Top 10

The 10 most critical security risks in LLM applications, with examples and mitigations.

## What you learn

- All 10 OWASP LLM vulnerabilities with concrete attack scenarios
- How each maps to traditional web security risks (and where they differ)
- Mitigations you can implement today
- How to score your application against each risk

## Run

```bash
python owasp.py
```

## OWASP LLM Top 10 (2025 edition)

| # | Risk | Core danger |
|---|---|---|
| LLM01 | Prompt Injection | User input hijacks LLM instructions |
| LLM02 | Insecure Output Handling | LLM output rendered without sanitization → XSS, SQLi |
| LLM03 | Training Data Poisoning | Backdoored training data manipulates inference |
| LLM04 | Model Denial of Service | Expensive prompts exhaust API budget |
| LLM05 | Supply Chain Vulnerabilities | Compromised models, plugins, datasets |
| LLM06 | Sensitive Info Disclosure | Model leaks system prompts, PII, training data |
| LLM07 | Insecure Plugin Design | Plugins with excessive permissions take unintended actions |
| LLM08 | Excessive Agency | Agent performs unintended destructive real-world actions |
| LLM09 | Overreliance | Users/devs trust hallucinated output without verification |
| LLM10 | Model Theft | Model weights or behavior extracted via API abuse |

## Quick wins (implement these first)

1. Never trust user input as instructions — separate system from user context
2. Sanitize LLM output before rendering in HTML (prevent XSS)
3. Token budget per user/session (prevent DoS)
4. Least-privilege for agent tools (prevent excessive agency)
5. Log everything — you need this for incident response
