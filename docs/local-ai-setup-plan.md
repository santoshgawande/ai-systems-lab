# Local AI Setup Plan — Multi-Model Lab

Goal: run multiple local models via Ollama on Mac Studio, then build CLI tools and scripts
that route, compare, and benchmark them — the same pattern used inside production AI gateways.

---

## What's Running Locally

All models served at `http://localhost:11434` via Ollama on Mac Studio M4 Max (64GB RAM).

| Model | Tag | Best For | Size |
|---|---|---|---|
| Llama 3.3 | `llama3.3:70b` | General tasks, balanced | ~40GB |
| DeepSeek R1 | `deepseek-r1` | Reasoning, math, logic | ~20GB |
| Qwen 2.5 Coder | `qwen2.5-coder:32b` | Code generation, debugging | ~20GB |
| GLM-4 Flash | `glm-4.7-flash` | Fast responses, simple tasks | ~5GB |
| Phi-4 | `phi4` | Small, fast, eval runner | ~9GB |
| Nomic Embed | `nomic-embed-text` | Embeddings for RAG | ~300MB |

Pull commands:
```bash
ollama pull llama3.3:70b
ollama pull deepseek-r1
ollama pull qwen2.5-coder:32b
ollama pull glm4:flash       # or glm-4.7-flash depending on tag
ollama pull phi4
ollama pull nomic-embed-text
```

---

## Multi-Model Patterns to Learn

### Pattern 1 — Compare (side-by-side)
Send the same prompt to N models, display outputs in parallel.
Use: spot quality differences, understand each model's style.

### Pattern 2 — Router (smart dispatch)
Classify the task type → send to the best model for that type.
Use: cost/latency optimization — don't burn a 70B model on a simple question.

### Pattern 3 — Fallback Chain
Try model A → if it fails/times out → fall back to model B.
Use: reliability — local model fails, route to API.

### Pattern 4 — Benchmark
Run the same prompt across models, measure latency + tokens/sec.
Use: understand trade-offs before choosing a model for a feature.

### Pattern 5 — Ensemble / Voting
Ask N models the same question, pick the most common answer.
Use: improve factual accuracy on classification/structured tasks.

---

## Lab Structure

```
01-llm-apis/
├── README.md
├── requirements.txt
├── 01-hello-ollama/        — basic call, see raw request/response
├── 02-streaming/           — stream tokens, understand SSE
├── 03-multi-model-compare/ — compare N models side by side (parallel)
├── 04-model-router/        — route to best model by task type
└── 05-benchmark/           — latency + throughput benchmark across models
```

---

## Setup

```bash
cd 01-llm-apis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# verify Ollama is running
curl http://localhost:11434/api/tags | python -m json.tool
```

---

## What Each Lab Teaches

| Lab | Core concept | Applies to |
|---|---|---|
| 01-hello-ollama | Raw HTTP API, request/response structure | Phase 1.1 |
| 02-streaming | SSE streaming, token-by-token output | Phase 1.3 |
| 03-multi-model-compare | Parallel requests, output comparison | Phase 4.2 (model routing) |
| 04-model-router | Task classification, routing logic | Phase 4.2 (cost optimization) |
| 05-benchmark | Latency, tokens/sec, p50/p95 | Phase 4.3 (observability) |

---

## Extending This Later

Once the gateway (LiteLLM on proxmox1) is up:
- Point these scripts at `http://proxmox1:4000` instead of `localhost:11434`
- Add cloud models (claude-sonnet, gpt-4o) to the compare/router scripts
- The router becomes the core of your mini-ai-gateway (Phase 6)
