import sys
import httpx
import psycopg2

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB = "host=proxmox1 port=5432 dbname=postgres user=postgres password=postgres"
TOP_K = 5


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG?"

print(f"Query: {query!r}")
print(f"Embedding with {EMBED_MODEL}...")
q_vec = embed(query)

conn = psycopg2.connect(DB)
cur = conn.cursor()

cur.execute("""
    SELECT
        content,
        source,
        chunk_index,
        1 - (embedding <=> %s::vector) AS similarity
    FROM rag_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s
""", (str(q_vec), str(q_vec), TOP_K))

results = cur.fetchall()
cur.close()
conn.close()

if not results:
    print("\nNo results. Run 02-indexing-pipeline/ingest.py first.")
    sys.exit(1)

print(f"\nTop {len(results)} results:")
print("=" * 70)
for i, (content, source, chunk_idx, sim) in enumerate(results, 1):
    bar = "█" * int(sim * 20)
    print(f"\n[{i}] sim={sim:.3f} {bar}  {source}:chunk{chunk_idx}")
    print("-" * 60)
    print(content[:400])
    if len(content) > 400:
        print("...")
