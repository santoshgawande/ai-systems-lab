# Lab 01 — Chunking Strategies

Chunking is the most important factor in RAG quality. See how different strategies affect chunk count, size, and boundary quality.

## What you learn

- **Fixed-size**: simple, predictable, but splits mid-sentence
- **Sentence**: preserves grammatical units, variable size
- **Paragraph**: natural topic boundaries, can produce very large chunks
- **Recursive**: tries paragraph → sentence → word boundaries in order — best general-purpose strategy

## Run

```bash
python chunk.py
```

## Key insight

Overlap is critical. Without overlap, an answer that spans two chunks will be missed.
A 20% overlap (e.g. 80 chars overlap on a 400-char chunk) is a good starting point.
