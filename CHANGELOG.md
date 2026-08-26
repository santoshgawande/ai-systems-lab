# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `06-build-your-own/rag-agent/` — the full RAG application, moved in from the
  standalone `rag-agent-lab` directory, which had never been placed under version
  control. Router agent, tool registry (RAG search, safe calculator, web search),
  and a Streamlit inspector that shows retrieved chunks with similarity scores.
  Runs on Docker with an embedded Chroma store and a bundled Ollama service, so it
  needs neither the homelab nor an API key.
- `30-speculative-decoding/` — Leviathan speculative sampling and Medusa multi-head tree attention.
- `31-paged-attention-kv-cache/` — vLLM physical block memory allocator and Radix-tree prefix KV caching.
- `32-mcp-protocol/` — Anthropic Model Context Protocol (MCP) JSON-RPC 2.0 server and multi-server client orchestrator.
- `33-alignment-dpo-rlhf/` — Direct Preference Optimization (DPO) loss engine and preference dataset pipeline.
- `34-reasoning-models/` — Test-Time Compute (TTC) thinking budget parser and Process Reward Model (PRM) step verifier.
- `35-mixture-of-experts/` — Sparse MoE Top-K gating router and auxiliary load balancing loss engine.
- Cross-links between the three RAG levels — `03-rag/` (stages in isolation),
  `06-build-your-own/mini-rag/` (one file) and `06-build-your-own/rag-agent/`
  (full app) — so the overlap reads as a deliberate progression.

### Changed

- `rag-agent`'s Compose file no longer requires a `.env`. `OLLAMA_BASE_URL`
  defaults to the Compose service hostname, which `app/config.py`'s `localhost`
  default gets wrong inside a container.

### Security

- `.gitignore` now excludes generated vector indexes (`**/data/chroma/`,
  `**/*.sqlite3`).
- The standalone project's `.env`, which held a live `ANTHROPIC_API_KEY`, was not
  carried across. Only `.env.example` moved.

## [0.1.0] — 2026-08-09

The labs had been written against the March scaffold but never committed. This
release puts all of them under version control for the first time.

### Added

- **Foundations (01–05)** — raw LLM API calls with streaming, vision and token
  counting; embeddings and cosine similarity over pgvector; a four-stage RAG
  pipeline (chunk → index → retrieve → generate); agents covering tool use, the
  ReAct loop, multi-agent orchestration and memory; and system design labs on
  reliability, cost, observability and gateways.
- **06-build-your-own** — six from-scratch systems: `mini-claude-code`,
  `mini-rag`, `mini-chatgpt`, `mini-copilot`, `mini-eval-framework` and
  `mini-ai-gateway`.
- **Technique labs** — prompt engineering, evals (including LLM-as-judge),
  guardrails, BM25 + RRF hybrid search, cross-encoder reranking with MMR,
  Instructor/Pydantic structured output, Whisper STT and TTS, document and table
  parsing, constitutional self-critique, and Hugging Face transformers with
  quantisation.
- **Provider labs** — Claude (prompt caching, extended thinking, MCP), OpenAI
  (function calling, structured outputs, Assistants), Gemini (long context,
  multimodal, grounding), and LiteLLM routing with fallbacks.
- **Infrastructure and security** — fine-tuning with Ollama Modelfiles and
  LoRA/QLoRA, Qdrant vs pgvector benchmarking, the OWASP LLM Top 10 with prompt
  injection and red-teaming labs, batch processing, and the Claude Code SDK.
- **Agent frontier** — orchestrator/subagent and agent-as-tool patterns, context
  management (sliding window, summarisation, human-in-the-loop), and an
  autonomous overnight task runner.
- `SECURITY-PLAN.md`, plus planning docs: weekend plan, resources checklist and
  the local multi-model Ollama setup plan.

### Changed

- `README.md` now documents all 28 sections with per-section lab counts, replacing
  the six-section outline from the initial scaffold.

### Security

- `.gitignore` now excludes scratch CV and resume reviews under `docs/`. This
  repository is public, and those documents concern real third parties.

## [0.0.1] — 2026-03-13

### Added

- Initial scaffold: README, LICENSE, learning plan and getting-started docs,
  homelab infrastructure notes, and Docker Compose deploy configs for proxmox1
  and proxmox2.
- `.gitignore` and `.env.example` for secret safety.
