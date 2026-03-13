# ai-systems-lab

Learning how production AI systems work — as a senior software engineer.
Each directory is a concept with working code and step-by-step instructions.

## Structure

```
ai-systems-lab/
├── docs/                        # learning plan, homelab setup, getting started
├── deploy/                      # proxmox1 + proxmox2 docker configs
│
├── 01-llm-apis/                 # raw API calls, streaming, token counting
├── 02-embeddings/               # vector embeddings, similarity, pgvector
├── 03-rag/                      # retrieval-augmented generation
├── 04-agents/                   # tool calling, ReAct loop
├── 05-system-design/            # gateway, observability, evals
└── 06-build-your-own/           # mini-claude-code, mini-rag, mini-gateway
```

## How to use

Each concept directory has:
- `README.md` — what you learn + steps to run
- numbered subdirectories — one per exercise, each with code + its own README

Start at `01-llm-apis/` and work forward.

## Docs

- [Getting started](docs/getting-started.md) — first steps + reading list
- [Learning plan](docs/learning-plan.md) — full topic checklist
- [Homelab infra](docs/homelab-infra.md) — Mac Studio + Proxmox setup
- [Deploy](deploy/README.md) — proxmox docker configs

## Homelab

| Machine | Role | Access |
|---|---|---|
| Mac Studio M4 Max 64GB | Ollama (LLMs + embeddings) | `localhost:11434` |
| proxmox1 (192.168.0.111) | PostgreSQL + pgvector, Redis | `proxmox1:5432`, `proxmox1:6379` |
| proxmox2 (192.168.0.112) | Jupyter Lab, Qdrant | `proxmox2:8888`, `proxmox2:6333` |
