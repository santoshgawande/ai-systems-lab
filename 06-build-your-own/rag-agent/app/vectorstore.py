"""ChromaDB-backed vector store with a local sentence-transformers embedder.

Embedded (on-disk) Chroma keeps the project to a single container — no separate
DB service to run. The embedder downloads its model once on first use.
"""
from __future__ import annotations

import functools

import chromadb
from sentence_transformers import SentenceTransformer

from .config import config


@functools.lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    # Cached so we load the (~80MB) model once per process.
    return SentenceTransformer(config.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    return _embedder().encode(texts, normalize_embeddings=True).tolist()


@functools.lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=config.chroma_dir)
    # We pass embeddings in ourselves, so no embedding_function on the collection.
    return client.get_or_create_collection(
        name=config.collection, metadata={"hnsw:space": "cosine"}
    )


def add(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    coll = _collection()
    coll.upsert(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metadatas)


def query(text: str, top_k: int | None = None) -> list[dict]:
    """Return the top_k most similar chunks with a 0..1 similarity score."""
    coll = _collection()
    k = top_k or config.top_k
    res = coll.query(query_embeddings=embed([text]), n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "chunk": meta.get("chunk", -1),
                # cosine distance -> similarity for a friendlier dashboard number
                "score": round(1 - dist, 3),
            }
        )
    return hits


def stats() -> dict:
    coll = _collection()
    return {"collection": config.collection, "chunks": coll.count()}
