#!/usr/bin/env python3
"""mini-rag: ingest documents and answer questions from them."""

import sys
import json
import hashlib
import httpx
import psycopg2
from psycopg2.extras import execute_values, Json
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.3:70b"
DB = "host=proxmox1 port=5432 dbname=postgres user=postgres password=postgres"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4

SAMPLE_DOC = """What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with LLM generation.
Instead of relying on the model's training data, RAG retrieves relevant documents and includes them in the prompt.

Why use RAG?

RAG lets you answer questions about private documents the model was never trained on.
It reduces hallucination by grounding answers in real retrieved text.
It is cheaper than fine-tuning and easier to update — just add documents to the index.

How does chunking affect quality?

Chunking strategy is the most impactful factor in RAG quality.
Fixed-size chunking is simple but splits sentences at arbitrary boundaries.
Sentence-based chunking preserves grammatical units.
Recursive chunking tries paragraph, then sentence, then word boundaries — best general purpose approach.
Overlap between chunks prevents answers from being cut at a boundary.

What is a vector database?

A vector database stores embeddings and enables fast approximate nearest-neighbor search.
pgvector adds vector storage and cosine similarity search to PostgreSQL.
Qdrant and Pinecone are purpose-built vector databases for scale.
The query is embedded with the same model as the documents, then compared by cosine similarity."""


def chunk(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text.strip()]
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                tail = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
                current = (tail + sep + part) if tail else part
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if c.strip()]
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def setup(conn):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mini_rag (
            id          SERIAL PRIMARY KEY,
            doc_id      TEXT,
            source      TEXT,
            chunk_index INT,
            content     TEXT,
            embedding   vector(768)
        )
    """)
    conn.commit()
    cur.close()


def ingest(path: Path, conn):
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
    chunks = chunk(text)
    print(f"  {path.name}: {len(text)} chars → {len(chunks)} chunks (doc_id={doc_id})")

    rows = []
    for i, c in enumerate(chunks):
        rows.append((doc_id, path.name, i, c, str(embed(c))))
        print(f"    [{i+1}/{len(chunks)}]", end="\r")

    cur = conn.cursor()
    cur.execute("DELETE FROM mini_rag WHERE doc_id = %s", (doc_id,))
    execute_values(cur, "INSERT INTO mini_rag (doc_id, source, chunk_index, content, embedding) VALUES %s", rows)
    conn.commit()
    cur.close()
    print(f"    ✓ {len(rows)} chunks indexed")


def retrieve(query: str, conn) -> list[dict]:
    q_vec = embed(query)
    cur = conn.cursor()
    cur.execute("""
        SELECT content, source, chunk_index, 1 - (embedding <=> %s::vector) AS score
        FROM mini_rag ORDER BY embedding <=> %s::vector LIMIT %s
    """, (str(q_vec), str(q_vec), TOP_K))
    rows = cur.fetchall()
    cur.close()
    return [{"content": r[0], "source": r[1], "chunk": r[2], "score": r[3]} for r in rows]


def generate(query: str, chunks: list[dict]):
    context = "\n\n---\n\n".join(
        f"[{c['source']} chunk {c['chunk']} | score={c['score']:.2f}]\n{c['content']}"
        for c in chunks
    )
    messages = [
        {"role": "system", "content": "Answer only from the provided context. Cite sources. Say 'not in documents' if unknown."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    print("\nAnswer:")
    print("-" * 60)
    with httpx.stream("POST", f"{OLLAMA}/api/chat",
                      json={"model": LLM_MODEL, "messages": messages, "stream": True},
                      timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                if not d.get("done"):
                    print(d["message"]["content"], end="", flush=True)
    print("\n")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py ingest <file_or_dir>")
        print("  python app.py query  <question>")
        sys.exit(1)

    cmd = sys.argv[1]
    conn = psycopg2.connect(DB)
    setup(conn)

    if cmd == "ingest":
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("sample.txt")
        if not target.exists() and target.name == "sample.txt":
            target.write_text(SAMPLE_DOC)
            print("Created sample.txt")

        files = sorted(target.glob("**/*.txt") if target.is_dir() else [target])
        files += sorted(target.glob("**/*.md")) if target.is_dir() else []
        print(f"Ingesting {len(files)} files...")
        for f in files:
            ingest(f, conn)
        print(f"\nDone. Run: python app.py query \"your question\"")

    elif cmd == "query":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Question: ")
        print(f"Query: {query!r}")
        chunks = retrieve(query, conn)
        if not chunks:
            print("No results. Run ingest first.")
            sys.exit(1)
        print(f"\nTop {len(chunks)} chunks:")
        for c in chunks:
            print(f"  [{c['score']:.3f}] {c['source']}:chunk{c['chunk']}  {c['content'][:60]}...")
        generate(query, chunks)

    conn.close()


if __name__ == "__main__":
    main()
