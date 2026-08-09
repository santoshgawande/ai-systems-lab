# Lab 04 — RAG Generation

Full pipeline: retrieve context → build prompt → stream LLM answer with source attribution.

## What you learn

- How retrieved chunks are injected into the system prompt
- Why you instruct the model to "answer only from context"
- How to attribute sources so the answer is verifiable
- The full end-to-end RAG pipeline in ~60 lines of Python

## Run

```bash
python generate.py "what is RAG?"
python generate.py "how does chunking affect retrieval?"
python generate.py "what is a vector database used for?"
```

## The prompt structure

```
System: You answer only from provided context. Cite sources.

Context:
[source: sample.txt chunk 0 | score=0.91]
RAG (Retrieval-Augmented Generation) is a technique...

---

[source: sample.txt chunk 2 | score=0.84]
Vector databases like pgvector, Qdrant...

Question: what is RAG?
```
