# 03 — RAG (Retrieval-Augmented Generation)

Ground LLM responses in real documents. The pattern behind Perplexity, ChatGPT file uploads, and Claude's knowledge retrieval.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires:
- Ollama at `http://localhost:11434`
- PostgreSQL + pgvector at `proxmox1:5432`

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-chunking/` | Fixed, sentence, paragraph, and recursive chunking strategies | `python chunk.py` |
| `02-indexing-pipeline/` | Full ingest: text → chunk → embed → store in pgvector | `python ingest.py sample.txt` |
| `03-retrieval/` | Semantic search, top-k results with similarity scores | `python retrieve.py "your question"` |
| `04-generation/` | Full RAG: retrieve context → format prompt → stream answer | `python generate.py "your question"` |

## The RAG pipeline

```
Document
  → Chunk (split into pieces)
  → Embed (each chunk → vector)
  → Store (vectors in pgvector)

Query
  → Embed (query → vector)
  → Search (cosine similarity against stored vectors)
  → Top-K chunks retrieved
  → Prompt = system + chunks + question
  → LLM generates answer grounded in retrieved context
```

## Key concepts

- Chunking strategy affects retrieval quality more than model choice — get this right first
- Overlap between chunks prevents answers from being cut at a chunk boundary
- The retrieval step is a nearest-neighbor search in vector space — not keyword matching
- Re-ranking (cross-encoder) improves precision over cosine similarity alone

## Where to go next

These four labs teach each stage in isolation. To see them composed:

- [`../06-build-your-own/mini-rag/`](../06-build-your-own/mini-rag/) — the whole
  pipeline in a single 187-line file
- [`../06-build-your-own/rag-agent/`](../06-build-your-own/rag-agent/) — a full app
  with a router agent and an inspector that shows retrieved chunks and their scores

Related techniques that plug into this pipeline live in
[`../18-hybrid-search/`](../18-hybrid-search/) (BM25 + RRF),
[`../21-reranking/`](../21-reranking/) (cross-encoder, MMR) and
[`../08-evals/`](../08-evals/) (measuring whether any of it helped).
