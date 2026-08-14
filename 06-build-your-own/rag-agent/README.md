# 🔎 rag-agent

A small, readable project for **learning how a RAG pipeline, a router-style AI
agent, and a live inspector dashboard fit together**. Everything runs in Docker
and works either fully offline (local Ollama model) or against the Anthropic
Claude API — toggle with one env var.

## Where this sits

This is the third and largest of three passes at the same idea. They are a
deliberate progression, not duplication — do not "clean up" the smaller ones:

| Level | Where | Size | What it gives you |
|---|---|---|---|
| Concepts | [`../../03-rag/`](../../03-rag/) | 328 lines | Each stage alone: chunk → index → retrieve → generate |
| One file | [`../mini-rag/`](../mini-rag/) | 187 lines | The whole pipeline end-to-end, nothing else |
| Full app | **here** | ~490 lines | Router agent, tool dispatch, Docker, and an inspector that makes every stage visible |

Read them in that order. The inspector is the reason this one exists: retrieval
quality is invisible until you can see which chunks were retrieved, at what
score, and which of them actually reached the answer prompt.

```
question ──> [Router agent] ──pick tool──> rag_search / calculator / web_search
                  │                                   │
                  └────────── tool output ────────────┘
                                  │
                                  ▼
                         [Answer agent] ──> grounded reply + trace
                                                   │
                                                   ▼
                                Streamlit dashboard (chat + 🔬 inspector)
```

## What's inside

| Piece            | File                  | What it teaches |
|------------------|-----------------------|-----------------|
| LLM abstraction  | `app/llm.py`          | One interface, two backends (Claude / Ollama) |
| Vector store     | `app/vectorstore.py`  | Embeddings + ChromaDB + cosine similarity |
| Ingestion        | `app/ingest.py`       | Chunking with overlap |
| Tools            | `app/tools.py`        | RAG search, safe calculator, web search |
| Router agent     | `app/agent.py`        | route → act → answer, with a full trace |
| Dashboard        | `app/dashboard.py`    | Chat + inspector showing retrieved chunks & routing |

## Quick start (Docker — recommended)

```bash
cd 06-build-your-own/rag-agent

docker compose up -d --build  # starts the app + an ollama service

# pull a small local model (one time)
docker compose exec ollama ollama pull llama3.2
```

No `.env` is needed to start — the defaults run fully offline against the bundled
Ollama service. Copy `.env.example` to `.env` only when you want to change the
backend or tune retrieval.

Open <http://localhost:8501>, click **📥 Ingest data/docs** in the sidebar, then
ask away:

- *"How does retrieval work in a RAG pipeline?"* → routes to `rag_search`
- *"What is 1299 * 1.18?"* → routes to `calculator`
- *"Who founded DuckDuckGo?"* → routes to `web_search`

The **🔬 Inspector** panel shows which tool the agent picked, why, and the exact
chunks (with similarity scores) that grounded the answer.

### Use Claude instead of local Ollama

In `.env`:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Then `docker compose up -d` again. No code changes needed.

## Things to try (learning exercises)

1. Change `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env`, re-ingest, and watch retrieval
   scores change in the inspector.
2. Add your own `.md`/`.txt` files to `data/docs/` and re-ingest.
3. Add a new tool in `app/tools.py` (e.g. a unit converter) and register it — the
   router and dashboard pick it up automatically.
4. Compare the same question on `ollama` vs `anthropic` and look at routing quality.

## Notes

- The vector store persists in `data/chroma/` (mounted volume), so your index
  survives restarts. It is gitignored — run the ingest step to rebuild it.
- Embeddings run locally via `sentence-transformers` regardless of LLM backend.
- **This project deliberately does not use the homelab.** The rest of the repo
  assumes pgvector on `proxmox1:5432` and a remote Ollama; this one is
  self-contained Docker + embedded Chroma so it runs from a clean clone on any
  machine. Keep it that way.
- `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-6`, which is stale. Set a current
  model id in `.env` when using the Anthropic backend — tracked as ASL-4.
