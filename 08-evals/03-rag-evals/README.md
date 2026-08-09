# Lab 03 — RAG Evals

Evaluate your RAG pipeline on three dimensions: faithfulness, relevance, and correctness.

## What you learn

- **Faithfulness**: does the answer only use information from the retrieved context? (measures hallucination)
- **Relevance**: are the retrieved chunks actually relevant to the question? (measures retrieval quality)
- **Correctness**: does the answer match the expected answer? (measures end-to-end quality)
- How to use a judge LLM to score each dimension automatically

## Run

```bash
# Run ingest first if rag_chunks table is empty
cd ../../03-rag/02-indexing-pipeline && python ingest.py && cd ../../08-evals/03-rag-evals

python rag_eval.py
```

## RAGAS-style scoring

This implements a simplified version of the RAGAS evaluation framework:

```
Faithfulness = fraction of answer claims that are supported by the context
Relevance    = fraction of retrieved chunks that address the question
Correctness  = does the answer match the reference answer? (1 or 0)
```

## How to use this in CI

1. Create a golden dataset: 20 question + expected_answer pairs
2. Run rag_eval.py on every prompt or pipeline change
3. Gate deployment on minimum scores (e.g., faithfulness > 0.85, correctness > 0.80)
