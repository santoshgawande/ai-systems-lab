# Lab 02 — Model-Graded Evals (LLM-as-Judge)

Use a strong model to grade another model's outputs. The pattern behind PromptFoo, Braintrust, and LangSmith.

## What you learn

- How to write a rubric that a judge LLM can apply consistently
- Why you use a stronger model (llama3.3:70b) to grade a weaker one (phi4)
- How to get a score (1-5) + reasoning from the judge
- The bias problem: judges tend to favor longer, more confident answers

## Run

```bash
python model_graded.py
```

## Judge prompt structure

```
System: You are an expert evaluator. Grade against the rubric.
        Return JSON: {"score": 1-5, "reasoning": "...", "verdict": "PASS|FAIL"}

User:   Question: {question}
        Response: {response}
        Rubric:   {rubric}
```

## When to use model-graded evals

- Open-ended responses (summaries, explanations, code reviews)
- Subjective quality measures (helpfulness, clarity, tone)
- Anything where you can't write a deterministic check function

## Limitations

- Judge can be wrong — run your judge's grades against a human-labeled sample
- Judge has positional bias — longer first option tends to win
- Same model judging itself introduces systematic bias
