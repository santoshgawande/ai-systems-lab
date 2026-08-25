# ai-systems-lab

Learning how production AI systems work — as a senior software engineer.
Each directory is a concept with working code and step-by-step instructions.

## Structure

```
ai-systems-lab/
├── docs/                        # learning plan, homelab setup, getting started
├── deploy/                      # proxmox1 + proxmox2 docker configs
│
├── 01-llm-apis/                 # raw API calls, streaming, vision, token counting (7 labs)
├── 02-embeddings/               # vector embeddings, cosine similarity, pgvector (3 labs)
├── 03-rag/                      # chunking, indexing, retrieval, RAG generation (4 labs)
├── 04-agents/                   # tool use, ReAct loop, multi-agent, memory (4 labs)
├── 05-system-design/            # reliability, cost, observability, gateway (4 labs)
├── 06-build-your-own/           # mini-claude-code, mini-rag, mini-chatgpt, mini-copilot, mini-eval, mini-ai-gateway (6 projects)
├── 07-prompt-engineering/       # few-shot, chain-of-thought, structured output, system prompt design (4 labs)
├── 08-evals/                    # unit evals, LLM-as-judge, RAG evals (3 labs)
├── 09-guardrails/               # input guards, output guards (2 labs)
│
├── 10-claude-api/               # Anthropic basics, prompt caching, extended thinking, MCP (4 labs)
├── 11-openai-api/               # function calling, structured outputs, Assistants API (3 labs)
├── 12-gemini-api/               # long context, multimodal, grounding (3 labs)
│
├── 13-fine-tuning/              # when to fine-tune, Ollama Modelfiles, LoRA/QLoRA (3 labs)
├── 14-vector-databases/         # Qdrant, pgvector vs Qdrant benchmark, metadata filtering (3 labs)
├── 15-ai-security/              # OWASP LLM Top 10, prompt injection, red teaming (3 labs)
├── 16-batch-processing/         # OpenAI Batch API, Anthropic batches, async queue (3 labs)
├── 17-claude-code-sdk/          # CLAUDE.md, hooks, MCP server development (3 labs)
├── 18-hybrid-search/            # BM25, RRF hybrid fusion (2 labs)
├── 19-instructor/               # Pydantic + LLMs, retry validation (2 labs)
├── 20-audio-tts/                # Whisper STT, OpenAI TTS (2 labs)
│
├── 21-reranking/                # cross-encoder reranking, MMR diversity, Cohere API, RRF, ColBERT MaxSim (5 labs)
├── 22-litellm/                  # unified LLM API, router with fallbacks (2 labs)
├── 23-document-parsing/         # PDF text extraction, table extraction (2 labs)
├── 24-constitutional-ai/        # self-critique loop, OpenAI moderation API (2 labs)
├── 25-huggingface/              # transformers pipeline, model quantisation (2 labs)
├── 26-multi-agent-patterns/     # orchestrator-subagent, agent-as-tool (2 labs)
├── 27-context-management/       # sliding window, summarisation, human-in-the-loop (3 labs)
├── 28-autonomous-runner/        # overnight task queue runner — uses Claude Code quota while you sleep
└── 29-graph-rag/                # Microsoft GraphRAG, knowledge graph extraction, hierarchical community search (2 labs)
```

## How to use

Each concept directory has:
- `README.md` — what you learn + steps to run
- numbered subdirectories — one per exercise, each with code + its own README

Start at `01-llm-apis/` and work forward.

## Docs

- [Weekend plan](docs/weekend-plan.md) — what to do this weekend (May 3–4)
- [Getting started](docs/getting-started.md) — first steps + reading list
- [Learning plan](docs/learning-plan.md) — full topic checklist
- [Resources checklist](docs/resources-checklist.md) — docs, courses, papers, and videos to check off
- [Local AI setup plan](docs/local-ai-setup-plan.md) — multi-model local setup with Ollama
- [Homelab infra](docs/homelab-infra.md) — Mac Studio + Proxmox setup
- [Deploy](deploy/README.md) — proxmox docker configs

## Homelab

| Machine | Role | Access |
|---|---|---|
| Mac Studio M4 Max 64GB | Ollama (LLMs + embeddings) | `localhost:11434` |
| proxmox1 (192.168.0.111) | PostgreSQL + pgvector, Redis | `proxmox1:5432`, `proxmox1:6379` |
| proxmox2 (192.168.0.112) | Jupyter Lab, Qdrant | `proxmox2:8888`, `proxmox2:6333` |
