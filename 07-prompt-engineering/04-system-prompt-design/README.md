# Lab 04 — System Prompt Design

The system prompt is a contract with the model. The more precise the contract, the more consistent the behavior.

## What you learn

- How weak/medium/strong system prompts affect response quality
- The four components of a strong system prompt: role + constraints + format + examples
- How to enforce output length, tone, and structure
- How to test system prompt changes without breaking existing behavior

## Run

```bash
python system_prompt.py
```

## The four components

```
Role:        "You are a senior Java developer with 10 years of production experience."
Constraints: "Never suggest frameworks the user didn't ask about. Be direct."
Format:      "Lead with the recommendation. Use code blocks. Max 200 words."
Examples:    Show one example Q&A that demonstrates the expected behavior.
```

Miss any of these and the model improvises — differently each run.

## System prompt as a test surface

Treat system prompts like code. When you change them:
1. Run your evals before and after
2. Compare output on your 20 most important inputs
3. Never deploy a prompt change without a regression check
