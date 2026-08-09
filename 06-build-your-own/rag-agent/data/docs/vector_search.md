# Vector Search and Embeddings

An embedding is a list of numbers (a vector) that represents the meaning of a
piece of text. Texts with similar meaning end up close together in vector space,
which is what makes semantic search possible.

## Similarity metrics
Cosine similarity measures the angle between two vectors and ignores their
magnitude, which is why it is the most common choice for text embeddings. A
cosine similarity of 1.0 means identical direction (very similar), while 0.0
means unrelated. Some libraries report cosine *distance* instead, where distance
= 1 - similarity, so a smaller distance means a closer match.

## Approximate nearest neighbor (ANN)
Searching every vector exactly is slow for large collections. ANN algorithms such
as HNSW trade a tiny amount of accuracy for a large speedup by organizing vectors
into a navigable graph. ChromaDB uses HNSW under the hood.

## Choosing an embedding model
Small models like all-MiniLM-L6-v2 (384 dimensions) are fast and run on a laptop,
making them ideal for learning. Larger models give better recall at the cost of
speed and memory. Whatever model you choose, you must embed both your documents
and your queries with the *same* model.
