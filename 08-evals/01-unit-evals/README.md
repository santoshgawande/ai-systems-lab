# Lab 01 — Unit Evals

Write assertions on LLM outputs the same way you write unit tests for code.

## What you learn

- How to express "expected behavior" as a callable check function
- How to run a batch of evals and get a pass/fail summary
- How to detect regressions when you change a system prompt
- What makes a good eval (specific, deterministic, fast)

## Run

```bash
python eval.py
```

## What a unit eval looks like

```python
Eval(
    name="positive_classification",
    system="Classify sentiment. Reply: POSITIVE, NEGATIVE, or NEUTRAL.",
    input="I love this product!",
    check=lambda output: "POSITIVE" in output.upper(),
    expected="Output contains POSITIVE"
)
```

## Good eval checklist

- [ ] Tests ONE specific behavior
- [ ] Check function is deterministic (no randomness in the check itself)
- [ ] Fast to run (use small model, short output)
- [ ] Covers the edge cases, not just the happy path
- [ ] Written BEFORE you write the prompt (TDD for prompts)
