# mini-eval-framework

A minimal LLM evaluation framework you can run in CI. Test prompts like software.

## What it teaches

- Eval dataset format: input + expected + check function
- Unit evals: assertion-based pass/fail
- Model-graded evals: LLM-as-judge for subjective quality
- Regression detection: fail the build when prompt changes break behavior
- How real eval frameworks (Braintrust, PromptFoo) work under the hood

## Run

```bash
python eval_framework.py              # run all evals
python eval_framework.py --ci        # exit 1 if any eval fails
python eval_framework.py --suite rag # run only the 'rag' suite
```

## Eval types

| Type | How it works | When to use |
|---|---|---|
| Exact match | `response == expected` | Deterministic outputs (JSON, code) |
| Contains | `expected in response` | Key phrases must appear |
| Regex | `re.search(pattern, response)` | Flexible format matching |
| Function | `check_fn(response)` | Custom logic |
| LLM judge | Judge model scores 1-5 | Subjective quality (helpfulness, accuracy) |

## CI integration

```yaml
# .github/workflows/eval.yml
- name: Run LLM evals
  run: python eval_framework.py --ci
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The build fails if any eval regresses — just like unit tests for code.
