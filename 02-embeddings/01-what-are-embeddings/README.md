# Lab 01 — What Are Embeddings?

Turn text into vectors and measure semantic similarity with cosine distance.

## What you learn

- How `nomic-embed-text` converts a sentence into a 768-dimensional vector
- Why semantically similar sentences produce nearby vectors
- How cosine similarity measures the angle between two vectors (not their magnitude)
- Why "The cat sat on the mat" and "A feline rested on a rug" score ~0.85 similarity

## Run

```bash
python embed.py
```

## Expected output

```
Embedding 6 sentences with nomic-embed-text...
Embedding dimensions: 768

Nearest neighbor for each sentence:
  'The cat sat on the mat.'
  → 'A feline rested on a rug.'  (similarity: 0.872)
  ...

Similarity matrix (all pairs):
     S1    S2    S3    S4    S5    S6
S1  1.00  0.87  0.42  0.38  0.41  0.29
...
```

## Key insight

Cosine similarity = 1.0 means vectors point in the same direction (identical meaning).
Cosine similarity = 0.0 means perpendicular (completely unrelated topics).
The magnitude of the vector doesn't matter — only the direction.
