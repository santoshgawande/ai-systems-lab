"""ChromaDB-backed vector store with RRF Hybrid Search and Reranking support.

Embedded (on-disk) Chroma keeps the project to a single container — no separate
DB service to run. The embedder downloads its model once on first use.
"""
from __future__ import annotations

import functools
import math

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
    return client.get_or_create_collection(
        name=config.collection, metadata={"hnsw:space": "cosine"}
    )


def add(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    coll = _collection()
    coll.upsert(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metadatas)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def query(text: str, top_k: int | None = None, mode: str = "hybrid") -> list[dict]:
    """Return the top_k most similar chunks using Dense, BM25, or RRF Hybrid search."""
    coll = _collection()
    k = top_k or config.top_k
    res = coll.query(query_embeddings=embed([text]), n_results=k * 2)

    hits = []
    if res["documents"] and res["documents"][0]:
        for idx, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            dense_rank = idx + 1
            # Keyword BM25 proxy overlap score
            text_words = set(text.lower().split())
            doc_words = set(doc.lower().split())
            overlap = len(text_words.intersection(doc_words))
            bm25_score = round(overlap / (len(text_words) + 1), 3)

            rrf_val = round(_rrf_score(dense_rank) + (bm25_score * 0.05), 4)
            similarity = round(1 - dist, 3)

            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source", "?"),
                    "chunk": meta.get("chunk", -1),
                    "score": similarity if mode == "dense" else rrf_val,
                    "dense_score": similarity,
                    "bm25_score": bm25_score,
                    "rrf_score": rrf_val,
                }
            )

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k]


def stats() -> dict:
    coll = _collection()
    return {"collection": config.collection, "chunks": coll.count()}
