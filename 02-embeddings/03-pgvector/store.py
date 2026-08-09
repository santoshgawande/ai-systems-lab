import httpx
import psycopg2
from psycopg2.extras import execute_values, Json

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB = "host=proxmox1 port=5432 dbname=postgres user=postgres password=postgres"

DOCUMENTS = [
    ("Python is a high-level, interpreted programming language.", {"topic": "python"}),
    ("Java is a statically typed, compiled programming language.", {"topic": "java"}),
    ("PostgreSQL is a powerful open-source relational database.", {"topic": "database"}),
    ("Redis is an in-memory data structure store used as a cache.", {"topic": "database"}),
    ("Docker containers package applications with their dependencies.", {"topic": "devops"}),
    ("Kubernetes orchestrates containerized applications at scale.", {"topic": "devops"}),
    ("Neural networks learn patterns from training data.", {"topic": "ml"}),
    ("Transformers use self-attention to process sequences.", {"topic": "ml"}),
    ("REST APIs communicate over HTTP using standard verbs.", {"topic": "api"}),
    ("GraphQL is a query language for APIs developed by Meta.", {"topic": "api"}),
]


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


conn = psycopg2.connect(DB)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("DROP TABLE IF EXISTS documents")
cur.execute("""
    CREATE TABLE documents (
        id       SERIAL PRIMARY KEY,
        content  TEXT,
        metadata JSONB,
        embedding vector(768)
    )
""")
conn.commit()
print("Created table: documents\n")

print(f"Embedding {len(DOCUMENTS)} documents...")
rows = []
for text, meta in DOCUMENTS:
    vec = embed(text)
    rows.append((text, Json(meta), str(vec)))
    print(f"  ✓ {text[:55]}")

execute_values(cur, "INSERT INTO documents (content, metadata, embedding) VALUES %s", rows)
conn.commit()
print(f"\nInserted {len(rows)} rows\n")

cur.execute("CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5)")
conn.commit()
print("Created IVFFlat index\n")

QUERIES = [
    "What should I use for caching?",
    "How do containers work in production?",
    "Tell me about machine learning models.",
]

for query in QUERIES:
    q_vec = embed(query)
    cur.execute("""
        SELECT content, metadata, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 3
    """, (str(q_vec), str(q_vec)))

    print(f"Query: {query!r}")
    for content, meta, sim in cur.fetchall():
        print(f"  {sim:.3f}  [{meta['topic']}]  {content}")
    print()

cur.close()
conn.close()
