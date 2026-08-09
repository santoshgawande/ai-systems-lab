# Lab 02 — Output Guards

Validate and filter LLM outputs before they reach your application or users.

## What you learn

- How to validate JSON output against a required schema
- How to detect hallucination signals ("as of my knowledge cutoff", confident-sounding invented facts)
- How to detect and filter unsafe content in model output
- The retry-on-failure pattern for output validation

## Run

```bash
python output_guard.py
```

## What to validate

| Check | Why |
|---|---|
| JSON schema valid | Your app will crash if the model wraps output in prose |
| Required keys present | Missing fields break downstream code silently |
| Hallucination signals | Model may confidently invent URLs, names, statistics |
| Unsafe content | Even with a safety-tuned model, edge cases slip through |
| Response length | Very short = refused; very long = probably off-rails |

## The retry pattern

```python
for attempt in range(max_retries):
    output = call_llm(prompt)
    result = validate(output)
    if result.ok:
        return output
    # Feed the validation error back as context
    prompt += f"\nError: {result.error}. Fix and retry."
```
