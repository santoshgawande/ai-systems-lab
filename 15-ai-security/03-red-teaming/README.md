# Lab 03 — Red Teaming

Systematically find vulnerabilities in your AI application before attackers do.

## What you learn

- Red teaming methodology: threat model → attack surface → test cases → findings
- Building an adversarial test suite that runs in CI
- Severity scoring: critical / high / medium / low for AI vulnerabilities
- Automated red teaming vs human red teaming — when each is needed

## Run

```bash
python red_team.py
# Uses Ollama locally — tests your local LLM app systematically
```

## Red team process

```
1. Threat model
   └─ Who are the attackers? (users, API callers, competing products)
   └─ What do they want? (leak data, bypass guardrails, abuse for free)
   └─ What access do they have? (API caller, chat UI, document uploads)

2. Attack surface
   └─ Every user input field
   └─ Uploaded documents (indirect injection)
   └─ System prompt (if leaked, enables targeted attacks)

3. Test cases
   └─ Direct injection variants (20+ patterns)
   └─ Jailbreak attempts (roleplay, hypotheticals, DAN)
   └─ Data extraction (PII, system prompt, internal state)
   └─ Resource abuse (long inputs, repeated calls)
   └─ Behavioral edge cases (conflicting instructions, ambiguous requests)

4. Findings
   └─ Critical: data exfiltration, RCE possible
   └─ High: guardrail bypass, competitor assistance
   └─ Medium: information disclosure (non-sensitive)
   └─ Low: inconsistent behavior, quality degradation
```

## Automated vs human red teaming

| Automated (this lab) | Human red teaming |
|---|---|
| Fast, repeatable, CI-friendly | Creative, novel attacks |
| Covers known patterns | Finds unknown unknowns |
| Cheap | Expensive |
| Run on every PR | Run quarterly or before launch |
| Good for regression | Good for discovery |

**Best practice:** automate what's known, hire humans for novel attack discovery.
