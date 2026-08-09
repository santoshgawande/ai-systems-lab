# 08 — Evals (Evaluation)

"Prompt changes break things silently." Evals are the test suite for AI systems.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Ollama at `http://localhost:11434`.

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-unit-evals/` | Write assertions on LLM outputs, run a pass/fail suite | `python eval.py` |
| `02-model-graded/` | LLM-as-judge: use a strong model to score another model's outputs | `python model_graded.py` |
| `03-rag-evals/` | Faithfulness, relevance, and answer correctness for RAG pipelines | `python rag_eval.py` |

## Why evals matter

You wouldn't ship code without tests. Don't ship prompts without evals.

A system prompt change that improves one case often silently breaks another.
Evals let you catch regressions before users do.

## Eval types

| Type | What it tests | When to use |
|---|---|---|
| Unit eval | Single input → expected output pattern | Formatting, classification, refusals |
| Model-graded | Quality, reasoning, helpfulness | Open-ended responses you can't match exactly |
| RAG eval | Faithfulness, relevance, completeness | Any retrieval pipeline |
| Human eval | Subjective quality | High-stakes decisions, final validation |
