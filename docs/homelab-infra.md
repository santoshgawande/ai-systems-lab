# Homelab Infrastructure — AI Systems Learning

## Hardware Inventory

| Machine | Role | Specs |
|---|---|---|
| **Mac Studio M4 Max** | Primary dev + local LLM inference | 64GB RAM, 1TB SSD |
| **Proxmox Node 1** | Services cluster (databases, observability, gateways) | Laptop 1 |
| **Proxmox Node 2** | AI workloads (vLLM, vector DBs, experiment runners) | Laptop 2 |

---

## What Goes Where

### Mac Studio M4 Max — Primary Dev Machine
Best for: local LLM inference (Apple Silicon = fast), coding, building

- **Ollama** — run Llama 3.3 70B, Mistral, Qwen2.5, Phi-4 locally (MLX-optimized)
- **Claude Code** — daily driver for AI-assisted development
- **VS Code / Cursor** — IDE
- **Python / Node.js dev environment** — all Phase 1–6 coding
- **Docker Desktop** — local quick experiments
- Models to run locally:
  - `llama3.3:70b` — general-purpose (fits in 64GB)
  - `qwen2.5-coder:32b` — code tasks
  - `nomic-embed-text` — local embeddings (free, fast)
  - `phi4` — small, fast for evals

---

### Proxmox Node 1 — Services Cluster

**VMs / LXC containers to run:**

| Service | Purpose | Learning Phase |
|---|---|---|
| **PostgreSQL + pgvector** | Vector store for RAG | Phase 2, 3 |
| **Redis** | Prompt cache, rate-limit state, session memory | Phase 3, 4 |
| **LiteLLM proxy** | AI gateway — multi-provider routing | Phase 4 |
| **Langfuse** (self-hosted) | LLM observability, traces, evals | Phase 4 |
| **Minio** | S3-compatible blob store for RAG docs | Phase 3 |
| **Nginx** | Reverse proxy / load balancer | Phase 4 |

**Suggested VM layout:**
```
proxmox-node-1/
├── vm-101: ubuntu-services (4 vCPU, 8GB RAM)
│   ├── postgres + pgvector
│   └── redis
├── vm-102: ai-gateway (2 vCPU, 4GB RAM)
│   ├── litellm proxy (port 4000)
│   └── nginx
└── vm-103: observability (4 vCPU, 8GB RAM)
    └── langfuse (docker-compose)
```

---

### Proxmox Node 2 — AI Workloads

**VMs / LXC containers to run:**

| Service | Purpose | Learning Phase |
|---|---|---|
| **vLLM** | Self-hosted OpenAI-compatible inference | Phase 5 |
| **Chroma / Qdrant** | Vector DBs (compare vs pgvector) | Phase 2, 3 |
| **Jupyter Lab** | Experiment notebooks, RAG prototyping | Phase 2–5 |
| **Airflow / Prefect** | RAG indexing pipelines | Phase 3 |
| **Braintrust / PromptFoo** | Eval runners | Phase 4 |

**Suggested VM layout:**
```
proxmox-node-2/
├── vm-201: inference (all CPU/RAM you can spare)
│   └── vLLM (serves Llama/Mistral via OpenAI API)
├── vm-202: vector-stores (4 vCPU, 8GB RAM)
│   ├── qdrant (port 6333)
│   └── chroma (port 8000)
└── vm-203: notebooks (4 vCPU, 8GB RAM)
    ├── jupyter lab (port 8888)
    └── eval runners
```

---

## Network Setup (Recommended)

```
Mac Studio (dev)
    │
    ├──[local network]── proxmox1 (192.168.0.111) — services cluster
    │
    └──[local network]── proxmox2 (192.168.0.112) — AI workloads

All on subnet: 192.168.0.0/24
Hostnames (add to /etc/hosts on Mac Studio):
  proxmox1         → 192.168.0.111   # services cluster
  proxmox2         → 192.168.0.112   # AI workloads
```

---

## Per-Phase Deployment Plan

### Phase 1 — API Foundations
- Mac Studio: install Ollama, pull models
- No Proxmox needed yet
- Cost: $0 (use free API tiers + local)

### Phase 2 — Core Concepts (Embeddings, Context)
- Node 1: spin up `postgres + pgvector` VM
- Mac Studio: dev against it from Python
- Node 2: spin up Qdrant for comparison

### Phase 3 — Patterns (RAG, Agents, Memory)
- Node 1: Redis (for memory/session state), Minio (doc store)
- Node 2: Jupyter for RAG experiments, Airflow for indexing pipelines
- Mac Studio: Ollama for local embeddings (nomic-embed-text)

### Phase 4 — System Design (Gateway, Observability, Evals)
- Node 1: LiteLLM proxy, Langfuse (this is the most important phase for infra)
- Node 2: PromptFoo or Braintrust for eval runners
- Mac Studio: point all dev code at gateway.lab instead of APIs directly

### Phase 5 — Advanced (Fine-tuning, Open-source Models)
- Node 2: vLLM (OpenAI-compatible, serves your fine-tuned models)
- Mac Studio: MLX for Apple Silicon fine-tuning (mlx-lm)
- Use vLLM + Mac Studio as two inference options to compare

### Phase 6 — Build Your Own
- Everything is already deployed — you just build on top of it
- mini-ai-gateway: runs on Node 1 alongside LiteLLM
- mini-rag: uses pgvector (Node 1) + Ollama (Mac Studio)
- mini-claude-code: runs on Mac Studio, can shell into Proxmox VMs

---

## Quick Start — First Things to Deploy

```bash
# 1. Mac Studio — install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.3:70b
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:32b

# 2. Proxmox Node 1 — postgres + pgvector (LXC or VM)
apt install postgresql-16 postgresql-16-pgvector
# then in psql:
CREATE EXTENSION vector;

# 3. Proxmox Node 1 — Redis
apt install redis-server

# 4. Proxmox Node 1 — LiteLLM (docker)
docker run -d -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/berriai/litellm:main \
  --config /app/config.yaml

# 5. Mac Studio — point dev code at local gateway
export OPENAI_BASE_URL=http://gateway.lab:4000
export OPENAI_API_KEY=sk-anything   # LiteLLM accepts any key
```

---

## Cost Estimate

| Resource | Cost |
|---|---|
| Proxmox (on existing laptops) | $0 |
| Ollama / local models | $0 |
| OpenAI API (dev/learning tier) | ~$5–20/month |
| Anthropic API (dev/learning tier) | ~$5–20/month |
| **Total** | **~$10–40/month** |

Running models locally on Mac Studio cuts API costs significantly for experimentation.

---

## Suggested OS / Stack

- **Proxmox VMs**: Ubuntu 24.04 LTS (best package support)
- **Containers**: Docker + docker-compose on each VM
- **Dev language**: Python 3.12 (best AI ecosystem support)
- **Secondary**: Node.js 22 (for streaming UI, SSE demos)
- **IaC**: Ansible playbooks to provision Proxmox VMs repeatably
