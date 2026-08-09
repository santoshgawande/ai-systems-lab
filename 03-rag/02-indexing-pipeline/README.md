# Lab 02 — Indexing Pipeline

Full ingest pipeline: read a text file → chunk → embed each chunk → store in pgvector.

## What you learn

- How to build a repeatable ingest pipeline
- How doc_id hashing enables re-ingestion without duplicates
- Why you bulk insert rather than insert one row at a time
- When to create the ANN index (after bulk insert, not before)

## Run

```bash
# Uses bundled sample.txt if no file provided
python ingest.py

# Or point at your own file
python ingest.py myfile.txt

# Or ingest a whole directory
python ingest.py ./docs
```

## Pipeline

```
File
  → read text
  → recursive chunk (400 chars, 80 overlap)
  → embed each chunk (nomic-embed-text)
  → bulk INSERT into rag_chunks table
  → IVFFlat index created after insert
```
