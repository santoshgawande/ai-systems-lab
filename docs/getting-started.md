# Getting Started

Hands-on steps and reading list to learn AI systems concepts using the homelab.

---

## Step 1 — Streaming API (30 mins)

Understand how every AI chat product streams responses.

```bash
pip install requests
```

**Basic call:**
```python
import requests

resp = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3",
    "prompt": "Explain what a transformer model is in 3 sentences",
    "stream": False
})
print(resp.json()["response"])
```

**Streaming (how ChatGPT/Claude actually work):**
```python
import requests, json

with requests.post("http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": "What is RAG?", "stream": True},
    stream=True) as r:
    for line in r.iter_lines():
        chunk = json.loads(line)
        print(chunk["response"], end="", flush=True)
```

What you learn: SSE streaming — how tokens flow from model to client in real time.

---

## Step 2 — Token Counting (20 mins)

Understand why AI APIs charge per token and why context limits matter.

```bash
pip install tiktoken
```

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you doing today?"
tokens = enc.encode(text)
print(f"Text:   {text}")
print(f"Tokens: {tokens}")
print(f"Count:  {len(tokens)}")
```

Try it with longer text and see how the count grows. This is what you pay for on every API call.

---

## Step 3 — Embeddings in Jupyter (1 hour)

Run this in Jupyter Lab at `http://proxmox2:8888`

Get your Mac Studio IP first:
```bash
ifconfig | grep "192.168"
```

Then in a notebook:

```python
import requests, numpy as np

MAC_STUDIO_IP = "192.168.0.XXX"  # replace with your actual IP

def embed(text):
    r = requests.post(f"http://{MAC_STUDIO_IP}:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text})
    return np.array(r.json()["embedding"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

e1 = embed("I love programming in Python")
e2 = embed("Python is my favorite language")
e3 = embed("I enjoy eating pizza")

print(f"Python vs Python: {cosine_similarity(e1, e2):.4f}")  # high ~0.9
print(f"Python vs pizza:  {cosine_similarity(e1, e3):.4f}")  # low  ~0.6
```

What you learn: embeddings convert text to numbers — similar meaning = similar numbers. This is the foundation of RAG.

---

## Step 4 — Store Embeddings in pgvector (1 hour)

Connect to postgres on proxmox1 and store vectors.

```bash
pip install psycopg2-binary
```

```python
import psycopg2, requests, numpy as np

MAC_STUDIO_IP = "192.168.0.XXX"

conn = psycopg2.connect(host="proxmox1", dbname="ailab", user="ai", password="ai")
cur = conn.cursor()

# Create table
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id serial PRIMARY KEY,
        content text,
        embedding vector(768)
    )
""")
conn.commit()

def embed(text):
    r = requests.post(f"http://{MAC_STUDIO_IP}:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text})
    return r.json()["embedding"]

# Store documents
docs = [
    "Python is a high-level programming language",
    "Redis is an in-memory key-value store",
    "PostgreSQL is a relational database",
    "Transformers are the architecture behind LLMs",
]

for doc in docs:
    vec = embed(doc)
    cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (doc, vec))
conn.commit()

# Search
query = "what database stores data in memory?"
query_vec = embed(query)

cur.execute("""
    SELECT content, embedding <-> %s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 2
""", (query_vec,))

print(f"Query: {query}\n")
for row in cur.fetchall():
    print(f"  {row[0]}  (distance: {row[1]:.4f})")
```

What you learn: vector similarity search — the retrieval part of RAG.

---

## Week-by-Week Plan

| Week | Build | Read |
|---|---|---|
| 1 | Steps 1 + 2 above | Karpathy LLM video + Anthropic prompt guide |
| 2 | Steps 3 + 4 above | RAG from scratch playlist + pgvector docs |
| 3 | Mini RAG — document Q&A over a PDF | OpenAI embeddings cookbook |
| 4 | Tool-calling agent | ReAct paper + Anthropic tool use docs |
| 5+ | mini-claude-code | AI Engineering book (Chip Huyen) |

---

## Reading List

### Week 1 — Foundations

| Resource | Link | Time |
|---|---|---|
| Intro to LLMs — Andrej Karpathy | https://www.youtube.com/watch?v=zjkBMFhNj_g | 1 hr |
| Ollama API reference | https://github.com/ollama/ollama/blob/main/docs/api.md | 30 min |
| Anthropic prompt engineering guide | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview | 1 hr |

### Week 2 — RAG + Embeddings

| Resource | Link | Time |
|---|---|---|
| What are embeddings — OpenAI cookbook | https://cookbook.openai.com/articles/what_are_embeddings | 30 min |
| RAG from scratch — LangChain playlist | https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x | 3 hrs |
| pgvector README | https://github.com/pgvector/pgvector | 20 min |

### Week 3 — Agents + Tool Use

| Resource | Link | Time |
|---|---|---|
| ReAct paper | https://arxiv.org/abs/2210.03629 | 1 hr |
| Anthropic tool use docs | https://docs.anthropic.com/en/docs/build-with-claude/tool-use | 1 hr |
| AI Engineering — Chip Huyen (book) | https://www.oreilly.com/library/view/ai-engineering/9781098166298/ | ongoing |

---

## Models Available on Mac Studio

| Model | Best For |
|---|---|
| `llama3` | General tasks, fast |
| `deepseek-r1` | Reasoning, step-by-step thinking |
| `qwen3-coder:30b` | Code generation |
| `glm-4.7-flash` | Fast responses |
| `nomic-embed-text` | Embeddings (RAG) |

All served at `http://localhost:11434` — use this IP from proxmox2 notebooks: your Mac Studio LAN IP.
