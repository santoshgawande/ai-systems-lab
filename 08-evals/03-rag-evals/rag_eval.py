"""
RAG pipeline evaluation: faithfulness, relevance, and answer correctness.
Simplified RAGAS-style scoring using an LLM judge.
"""
import json
import httpx
import psycopg2

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.3:70b"
JUDGE_MODEL = "llama3.3:70b"
DB = "host=proxmox1 port=5432 dbname=postgres user=postgres password=postgres"
TOP_K = 4

RAG_SYSTEM = "Answer only from the provided context. If the answer isn't in the context, say 'not found in documents'."

GOLDEN_DATASET = [
    {
        "question": "What is RAG?",
        "expected": "RAG is Retrieval-Augmented Generation — a technique that retrieves documents and includes them in the LLM prompt.",
    },
    {
        "question": "How does chunking affect RAG quality?",
        "expected": "Chunking strategy is the most impactful factor in RAG quality. Overlap between chunks prevents answers from being cut at boundaries.",
    },
    {
        "question": "What is a vector database used for?",
        "expected": "Vector databases store embeddings and enable fast approximate nearest-neighbor search using cosine similarity.",
    },
]


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve(query: str, conn) -> list[dict]:
    q_vec = embed(query)
    cur = conn.cursor()
    cur.execute("""
        SELECT content, source, 1 - (embedding <=> %s::vector) AS score
        FROM rag_chunks ORDER BY embedding <=> %s::vector LIMIT %s
    """, (str(q_vec), str(q_vec), TOP_K))
    rows = cur.fetchall()
    cur.close()
    return [{"content": r[0], "source": r[1], "score": r[2]} for r in rows]


def generate(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(f"[{c['source']} | {c['score']:.2f}]\n{c['content']}" for c in chunks)
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        "stream": False,
    }, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def judge_score(prompt: str) -> dict:
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": 'Grade on a 0.0-1.0 scale. Respond ONLY with JSON: {"score": 0.0-1.0, "reason": "brief"}'},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }, timeout=60)
    r.raise_for_status()
    text = r.json()["message"]["content"].strip()
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e]) if s >= 0 else {"score": 0.0, "reason": "parse error"}
    except Exception:
        return {"score": 0.0, "reason": "parse error"}


def eval_faithfulness(answer: str, context: str) -> dict:
    return judge_score(
        f"Does the answer use ONLY information from the context (no hallucination)?\n\n"
        f"Context:\n{context[:1000]}\n\nAnswer:\n{answer}\n\n"
        f"Score 1.0 if fully grounded, 0.5 if mixed, 0.0 if hallucinated."
    )


def eval_relevance(question: str, chunks: list[dict]) -> dict:
    chunks_text = "\n".join(f"- {c['content'][:100]}" for c in chunks)
    return judge_score(
        f"Are these retrieved chunks relevant to the question?\n\n"
        f"Question: {question}\n\nChunks:\n{chunks_text}\n\n"
        f"Score 1.0 if highly relevant, 0.5 if partially relevant, 0.0 if irrelevant."
    )


def eval_correctness(answer: str, expected: str) -> dict:
    return judge_score(
        f"Does the answer convey the same meaning as the expected answer?\n\n"
        f"Expected: {expected}\n\nActual: {answer}\n\n"
        f"Score 1.0 if semantically equivalent, 0.5 if partially correct, 0.0 if wrong."
    )


try:
    conn = psycopg2.connect(DB)
except Exception as e:
    print(f"DB connection failed: {e}")
    print("Run: cd ../../03-rag/02-indexing-pipeline && python ingest.py")
    exit(1)

print(f"RAG Evaluation — {len(GOLDEN_DATASET)} test cases\n")
print(f"{'Question':<45} {'Faith':>6} {'Relev':>6} {'Corr':>6}  Notes")
print("-" * 80)

totals = {"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0}

for case in GOLDEN_DATASET:
    q = case["question"]
    chunks = retrieve(q, conn)
    answer = generate(q, chunks)
    context = "\n".join(c["content"] for c in chunks)

    faith = eval_faithfulness(answer, context)
    relev = eval_relevance(q, chunks)
    corr  = eval_correctness(answer, case["expected"])

    totals["faithfulness"] += faith["score"]
    totals["relevance"]    += relev["score"]
    totals["correctness"]  += corr["score"]

    print(f"  {q[:43]:<43}  {faith['score']:.2f}   {relev['score']:.2f}   {corr['score']:.2f}  {corr['reason'][:30]}")

conn.close()
n = len(GOLDEN_DATASET)
print("-" * 80)
print(f"  {'Average':<43}  {totals['faithfulness']/n:.2f}   {totals['relevance']/n:.2f}   {totals['correctness']/n:.2f}")
print(f"\nFaithfulness: measures hallucination (1.0 = fully grounded in context)")
print(f"Relevance:    measures retrieval quality (1.0 = retrieved chunks all relevant)")
print(f"Correctness:  measures end-to-end quality (1.0 = answer matches expected)")
