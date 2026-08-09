import sys
import hashlib
import httpx
import psycopg2
from psycopg2.extras import execute_values, Json
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB = "host=proxmox1 port=5432 dbname=postgres user=postgres password=postgres"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

SAMPLE = """RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language generation.

Instead of relying solely on a model's training data, RAG systems retrieve relevant documents from a knowledge base and include them in the prompt. This grounds the model's response in real, up-to-date information.

The key components of a RAG system are: document chunking, embedding generation, vector storage, similarity retrieval, and response generation. Each component affects overall quality.

Chunking strategy is often the most impactful factor. Fixed-size chunks are simple but may split sentences. Semantic chunking preserves meaning but is more complex. Recursive chunking is the best general-purpose approach.

Vector databases like pgvector, Qdrant, and Pinecone store embeddings and support fast approximate nearest-neighbor (ANN) search. The query embedding is compared against stored embeddings using cosine similarity.

Re-ranking improves precision by applying a cross-encoder model to the top-K retrieved chunks. This is slower than cosine similarity but more accurate."""


def chunk_recursive(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                tail = current[-overlap:] if len(current) > overlap else current
                current = (tail + sep + part) if tail else part
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if c]
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def setup(conn):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id          SERIAL PRIMARY KEY,
            doc_id      TEXT,
            source      TEXT,
            chunk_index INT,
            content     TEXT,
            metadata    JSONB,
            embedding   vector(768)
        )
    """)
    conn.commit()
    cur.close()


def ingest_file(path: Path, conn) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
    chunks = chunk_recursive(text)

    print(f"\nFile: {path.name}  ({len(text)} chars → {len(chunks)} chunks, doc_id={doc_id})")

    rows = []
    for i, c in enumerate(chunks):
        vec = embed(c)
        rows.append((doc_id, path.name, i, c, Json({"source": path.name, "chunk": i}), str(vec)))
        print(f"  [{i + 1:3d}/{len(chunks)}] {len(c)} chars", end="\r")

    cur = conn.cursor()
    cur.execute("DELETE FROM rag_chunks WHERE doc_id = %s", (doc_id,))
    execute_values(cur, """
        INSERT INTO rag_chunks (doc_id, source, chunk_index, content, metadata, embedding)
        VALUES %s
    """, rows)
    conn.commit()
    cur.close()
    print(f"\n  ✓ Inserted {len(rows)} chunks")
    return len(rows)


source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.txt")
if not source.exists() and source.name == "sample.txt":
    source.write_text(SAMPLE)
    print(f"Created sample.txt")

conn = psycopg2.connect(DB)
setup(conn)
print("Database ready.")

total = 0
if source.is_dir():
    files = list(source.glob("**/*.txt")) + list(source.glob("**/*.md"))
    for f in files:
        total += ingest_file(f, conn)
else:
    total += ingest_file(source, conn)

conn.close()
print(f"\nDone. Total chunks indexed: {total}")
print("Next: run 03-retrieval/retrieve.py to query")
