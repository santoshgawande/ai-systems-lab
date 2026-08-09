# Lab 03 — Retrieval

Query pgvector with an embedded question and retrieve the most relevant chunks.

## What you learn

- How the query goes through the same embedding model as the documents
- Why similarity scores matter (0.85+ is a strong match, below 0.5 is noise)
- How to interpret `<=>` (cosine distance) vs similarity (1 - distance)

## Run

```bash
# Run ingest first if you haven't
cd ../02-indexing-pipeline && python ingest.py && cd ../03-retrieval

python retrieve.py "what is RAG?"
python retrieve.py "how does chunking affect retrieval quality?"
python retrieve.py "what is a vector database?"
```
