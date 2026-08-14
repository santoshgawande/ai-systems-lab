"""Load documents from data/docs, chunk them, and upsert into the vector store."""
from __future__ import annotations

import pathlib

from . import vectorstore
from .config import config

DOCS_DIR = pathlib.Path("data/docs")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Word-based sliding window. Simple and good enough to learn the mechanics."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += step
    return chunks


def ingest_path(path: pathlib.Path) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
    if not chunks:
        return 0
    ids = [f"{path.name}:{i}" for i in range(len(chunks))]
    metas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]
    vectorstore.add(ids, chunks, metas)
    return len(chunks)


def ingest_all() -> dict:
    """Ingest every .txt/.md file in data/docs. Returns per-file chunk counts."""
    report = {}
    for path in sorted(DOCS_DIR.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            report[path.name] = ingest_path(path)
    return report


if __name__ == "__main__":
    print("Ingesting documents from", DOCS_DIR.resolve())
    for name, n in ingest_all().items():
        print(f"  {name}: {n} chunks")
    print("Store stats:", vectorstore.stats())
