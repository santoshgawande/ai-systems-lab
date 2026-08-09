import sys
import json
import httpx
import psycopg2

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.3:70b"
DB = "host=proxmox1 port=5432 dbname=postgres user=postgres password=postgres"
TOP_K = 4

SYSTEM = """You are a helpful assistant that answers questions strictly from the provided context.

Rules:
- Answer ONLY from the context below
- Cite which source each fact comes from
- If the context does not contain the answer, say: "I don't have that information in the provided documents"
- Be concise"""


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve(query: str, conn) -> list[dict]:
    q_vec = embed(query)
    cur = conn.cursor()
    cur.execute("""
        SELECT content, source, chunk_index, 1 - (embedding <=> %s::vector) AS score
        FROM rag_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (str(q_vec), str(q_vec), TOP_K))
    rows = cur.fetchall()
    cur.close()
    return [{"content": r[0], "source": r[1], "chunk": r[2], "score": r[3]} for r in rows]


def generate(query: str, chunks: list[dict]) -> None:
    context = "\n\n---\n\n".join(
        f"[source: {c['source']} chunk {c['chunk']} | score={c['score']:.2f}]\n{c['content']}"
        for c in chunks
    )
    messages = [
        {"role": "system", "content": SYSTEM},
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


query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG and how does it work?"

print(f"Question: {query}")
print(f"Retrieving top-{TOP_K} chunks from pgvector...")

conn = psycopg2.connect(DB)
chunks = retrieve(query, conn)
conn.close()

if not chunks:
    print("No chunks found. Run 02-indexing-pipeline/ingest.py first.")
    sys.exit(1)

print(f"\nRetrieved {len(chunks)} chunks:")
for c in chunks:
    print(f"  [{c['score']:.3f}] {c['source']}:chunk{c['chunk']}  {c['content'][:60]}...")

generate(query, chunks)
