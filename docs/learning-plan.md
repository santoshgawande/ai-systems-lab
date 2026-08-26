# AI Systems Learning Plan — Senior Software Engineer

> Goal: Understand how production AI systems (Claude Code, ChatGPT, Gemini, Copilot, etc.)
> are designed, built, and operated — from API to infra to evaluation.

---

## How to Use This Plan

- Work through phases **in order** — each builds on the last
- Each topic has a **check** — the thing you should be able to do/explain after learning it
- Mark topics `[x]` as you complete them
- Build something in every phase — reading alone won't make it stick
- Deep dives and conceptual interview answers: [`docs/mental-models-guide.md`](file:///Users/santosh/workspace/github/ai-systems-lab/docs/mental-models-guide.md)

---

## Phase 1 — LLM API Foundations
> Get fluent with raw APIs before learning patterns built on top of them.

### 1.1 API Basics
- [x] **OpenAI API** — completions, chat format, models, auth, pricing
  - Check: call GPT-4o, inspect the request/response JSON, understand `messages` array
- [x] **Anthropic API** — system prompts, `messages` format, model tiers
  - Check: call claude-sonnet-4-6, use a system prompt, understand how it differs from OpenAI
- [x] **Gemini API** — multimodal inputs, long context (1M tokens), grounding
  - Check: send an image + text prompt, observe the response

### 1.2 Token Mechanics
- [x] **Tokenization** — how text maps to tokens (BPE, tiktoken, sentencepiece)
  - Check: tokenize a sentence, count tokens manually, explain why "ChatGPT" is 2 tokens
- [x] **Context window** — prompt tokens + completion tokens = max window
  - Check: hit a context limit, handle the error, implement chunking
- [x] **Pricing model** — input vs output token costs, why output is more expensive
  - Check: estimate cost of 1000 API calls for a given use case

### 1.3 Streaming
- [x] **SSE (Server-Sent Events)** — how LLMs stream tokens to client
  - Check: implement a streaming client in Python or Node.js, render tokens as they arrive
- [x] **Backpressure** — what happens if the client can't keep up
  - Check: explain how SSE handles slow consumers vs WebSockets

---

## Phase 2 — Core Engineering Concepts
> The primitives all AI systems are built from.

### 2.1 Embeddings
- [x] **What embeddings are** — dense vector representation of text
  - Check: embed 10 sentences, compute cosine similarity, find nearest neighbors
- [x] **Embedding models** — text-embedding-3-small, voyage-3, Gemini embeddings
  - Check: compare embedding dimensions and quality trade-offs
- [x] **Vector stores** — pgvector, Pinecone, Weaviate, Chroma, Qdrant
  - Check: store 1000 embeddings in pgvector, run a similarity query

### 2.2 Prompt Engineering (Engineering, Not Magic)
- [x] **Few-shot prompting** — examples in context improve outputs
- [x] **Chain-of-thought** — step-by-step reasoning improves accuracy
- [x] **Structured output** — JSON mode, response_format, constrained decoding
  - Check: make an LLM always return valid JSON, handle parse failures
- [x] **System prompt design** — role, constraints, output format, examples
  - Check: write a system prompt that reliably produces consistent behavior

### 2.3 Context Management
- [x] **Sliding window** — trim old messages to fit context
- [x] **Summarization** — compress old turns into a summary
- [x] **Selective retrieval** — only include relevant history
  - Check: implement a multi-turn chat that stays within a 4k token budget

---

## Phase 3 — Patterns (The Core of All AI Products)
> Every real AI product is one or more of these patterns composed together.

### 3.1 RAG — Retrieval-Augmented Generation
- [x] **Chunking strategies** — fixed size, sentence, semantic, recursive
- [x] **Indexing pipeline** — ingest → chunk → embed → store
- [x] **Retrieval** — similarity search, MMR, hybrid (BM25 + vector)
- [x] **Generation** — stuff chunks into prompt, instruct model to answer from context
- [x] **Re-ranking** — cohere rerank, cross-encoder to improve top-k
  - Check: build a document Q&A pipeline over a PDF from scratch

### 3.2 Tool Use / Function Calling
- [x] **OpenAI function calling** — define tools as JSON schema, parse model's tool calls
- [x] **Anthropic tool use** — same pattern, different API shape
- [x] **Parallel tool calls** — model calls multiple tools at once
- [x] **Tool result handling** — feed results back into context
  - Check: build an agent that can call a weather API + calculator + search engine

### 3.3 Agentic Loops
- [x] **ReAct pattern** — Reason + Act: think → tool call → observe → repeat
- [x] **Plan-and-Execute** — plan all steps upfront, execute each
- [x] **Agentic loop termination** — stop conditions, max iterations, error handling
- [x] **Human-in-the-loop** — when to pause and ask for confirmation
  - Check: build a CLI agent that can read/write files and run shell commands

### 3.4 Memory Systems
- [x] **In-context memory** — conversation history in the prompt (ephemeral)
- [x] **External memory** — vector DB, key-value store (persistent)
- [x] **Episodic memory** — store and retrieve past interactions
- [x] **Semantic memory** — facts and knowledge the agent knows about a user
  - Check: build a chatbot that remembers user preferences across sessions

### 3.5 Prompt Caching
- [x] **Anthropic prompt caching** — cache prefix tokens, save cost on repeated system prompts
- [x] **OpenAI prompt caching** — automatic caching for long prompts
  - Check: measure cache hit rate and cost savings on a real workload

### 3.6 Multi-Agent Systems
- [x] **Orchestrator + subagent pattern** — one agent delegates to specialized agents
- [x] **Message-passing** — agents communicate via structured messages
- [x] **Shared context / blackboard** — agents share state via a central store
- [x] **Agent-as-tool** — one agent calls another as if it were a function
  - Check: build a 2-agent system (planner + executor) for a coding task

---

## Phase 4 — System Design for AI
> How to build production-grade systems around LLMs.

### 4.1 Reliability
- [x] **Rate limit handling** — exponential backoff, jitter, queue with token bucket
- [x] **Retry strategy** — which errors to retry (429, 503) vs not (400, 401)
- [x] **Fallback routing** — if primary model fails, fall back to secondary
- [x] **Timeout handling** — streaming timeouts, partial response recovery
  - Check: implement a resilient LLM client with retry + fallback

### 4.2 Cost Optimization
- [x] **Token budgeting** — limit output tokens, compress prompts
- [x] **Model routing** — use cheap model (Haiku) for simple tasks, expensive (Opus) for hard ones
- [x] **Caching** — cache identical or near-identical prompts
- [x] **Batching** — batch small requests to reduce overhead
  - Check: reduce per-call cost by 50% on a sample workload using these techniques

### 4.3 Observability
- [x] **LLM tracing** — log every prompt + completion (LangSmith, Arize, Braintrust)
- [x] **Token usage metrics** — track input/output tokens per request, per user, per day
- [x] **Latency tracking** — time-to-first-token (TTFT), total latency, p50/p95/p99
- [x] **Error tracking** — classify errors (model, rate limit, timeout, parse)
  - Check: instrument an LLM app so you can debug any failure from logs alone

### 4.4 AI Gateway / LLM Proxy
- [x] **What a gateway does** — auth, routing, rate limiting, logging, caching in one layer
- [x] **Open source options** — LiteLLM, Portkey, OpenRouter
- [x] **Build your own** — reverse proxy that adds auth + logging + fallback
  - Check: route requests to 3 different LLM providers through a single endpoint

### 4.5 Evaluation (Evals)
- [x] **Why evals matter** — "prompt changes break things silently"
- [x] **Types of evals** — unit evals, model-graded, human eval, A/B test
- [x] **Eval frameworks** — Braintrust, PromptFoo, RAGAS (for RAG)
- [x] **Regression testing** — run evals on every prompt change in CI
  - Check: write 20 evals for a RAG pipeline, run them in CI

### 4.6 Safety & Guardrails
- [x] **Input guardrails** — detect prompt injection, jailbreaks, PII
- [x] **Output guardrails** — validate JSON, detect hallucinations, filter harmful content
- [x] **Constitutional AI** — self-critique + revision
- [x] **Moderations API** — OpenAI moderation, Anthropic's usage policies
  - Check: add an input + output guard to an existing agent

---

## Phase 5 — Advanced Topics
> For going deep beyond standard patterns.

### 5.1 Fine-Tuning vs Prompting
- [x] **When to fine-tune** — consistent style/format, domain-specific knowledge, cost reduction
- [x] **When to prompt** — flexibility, fast iteration, small datasets
- [x] **Fine-tuning OpenAI** — JSONL dataset, training job, evaluation
- [x] **LoRA / QLoRA** — parameter-efficient fine-tuning for open-source models
  - Check: fine-tune a small model for a classification task, compare vs few-shot

### 5.2 Multimodal Systems
- [x] **Vision** — GPT-4o vision, Claude's vision, Gemini — image + text inputs
- [x] **Audio** — Whisper (STT), TTS, voice agents
- [x] **Document parsing** — PDF, tables, structured extraction
  - Check: build a pipeline that extracts data from a scanned invoice image

### 5.3 Structured Generation
- [x] **JSON mode** — guaranteed valid JSON output
- [x] **Pydantic + instructor** — parse LLM output directly into typed objects
- [x] **Constrained decoding** — Outlines, guidance — token-level control
  - Check: extract structured data from 100 free-text records reliably

### 5.4 Long Context Strategies
- [x] **Lost-in-the-middle problem** — models miss info in middle of long contexts
- [x] **Positional encoding limits** — why context degrades beyond training length
- [x] **Chunking + stitching** — process long docs in sections, merge results
  - Check: summarize a 200-page document accurately without losing key details

### 5.5 Open-Source Models
- [x] **Ollama / vLLM** — run Llama 3, Mistral, Qwen locally
- [x] **HuggingFace Transformers** — load, run, and understand open models
- [x] **Quantization** — GGUF, GPTQ, AWQ — run large models on consumer hardware
  - Check: run a 7B model locally, compare quality and speed vs GPT-4o-mini

---

## Phase 6 — Build Your Own (Internalize by Building)
> The highest-leverage learning. Build simplified versions of real AI systems.

| Project | What It Teaches | Status |
|---|---|---|
| `mini-claude-code` | Agentic loop, tool use, file I/O, streaming CLI | [x] Built & Tested |
| `mini-chatgpt` | Multi-turn state, memory, streaming UI | [x] Built & Tested |
| `mini-rag` | Full RAG pipeline: ingest → chunk → embed → retrieve → generate | [x] Built & Tested |
| `mini-copilot` | Code completion, context injection, IDE extension basics | [x] Built & Tested |
| `mini-eval-framework` | How to test AI systems like software | [x] Built & Tested |
| `mini-ai-gateway` | LLM proxy with auth, routing, logging, fallback | [x] Built & Tested |
| `rag-agent` | Full Dockerized RAG agent with Streamlit inspector | [x] Built & Tested |

---

## Phase 7 — Frontier Production AI Systems (2025–2026 Standards)
> The latest production techniques powering next-generation models and infrastructure.

### 7.1 Speculative Decoding
- [x] **Speculative Sampling** — draft-target rejection sampling, speedup math, zero quality degradation
- [x] **Medusa Multi-Head Decoding** — multiple heads on single base model, tree attention, candidate branches

### 7.2 PagedAttention & KV-Cache Management
- [x] **Physical Block Allocator** — virtual memory paging for KV-cache, zero memory fragmentation
- [x] **Radix Prefix Caching** — prefix tree matching, sub-millisecond TTFT, Copy-on-Write sharing

### 7.3 Model Context Protocol (MCP)
- [x] **MCP Server** — JSON-RPC 2.0 protocol, capability negotiation, tool & resource registration
- [x] **MCP Client Orchestrator** — multi-server discovery, schema translation, dynamic tool dispatch

### 7.4 Alignment & Direct Preference Optimization
- [x] **DPO Loss & Implicit Rewards** — analytic RLHF solution, reward margin calculation, beta temperature
- [x] **Preference Dataset Pipeline** — pairwise generation, verbosity length-bias mitigation, margin filtering

### 7.5 Reasoning Models & Test-Time Compute
- [x] **Test-Time Compute (TTC)** — dynamic thinking budget scaling, `<think>` trace parsing, self-correction
- [x] **Process Reward Model (PRM)** — step-level supervision, early error localization, Best-of-N search

### 7.6 Sparse Mixture of Experts (MoE)
- [x] **Top-K Gating Router** — sparse routing, softmax normalization, DeepSeek shared expert isolation
- [x] **Auxiliary Load Balancing Loss** — routing collapse mitigation, expert capacity limits, token dropping

---

## Phase 8 — Next-Gen Scaling, Attention & Memory Architectures
> Advanced architectural innovations powering 2026 flagship multimodal & long-context models.

### 8.1 FlashAttention & IO-Aware Tiling
- [x] **Online Softmax Algorithm** — 1-pass streaming softmax with dynamic rescaling
- [x] **SRAM Block Tiling** — FlashAttention-2 on-chip computation, zero $N \times N$ HBM writes

### 8.2 Multi-Head Latent Attention (DeepSeek MLA)
- [x] **Latent KV Compression** — low-rank joint KV compression, decoupled RoPE keys, 93.3% VRAM savings
- [x] **Inference Matrix Absorption** — direct query-latent projection without key decompression

### 8.3 State Space Models & Mamba
- [x] **Selective Scan Discretization** — input-dependent $\Delta(x_t)$ gating, Zero-Order Hold (ZOH)
- [x] **Constant $O(1)$ Memory Inference** — zero KV-cache overhead, linear $O(N)$ training

### 8.4 Multi-Token Prediction (MTP)
- [x] **Multi-Token Prediction Heads** — simultaneous lookahead prediction ($x_{t+1} \dots x_{t+M}$)
- [x] **Native Self-Speculative Decoding** — zero-overhead built-in draft proposals and verification

### 8.5 Diffusion Transformers (DiT)
- [x] **DiT Patchification & AdaLN-Zero** — 2D spatial grid to token sequence, timestep modulation
- [x] **Classifier-Free Guidance (CFG)** — trajectory vector extrapolation, reverse Euler ODE steps

### 8.6 Distributed Parallelism (Megatron-LM)
- [x] **Column-Row Tensor Parallelism** — weight slicing, single All-Reduce barrier per MLP
- [x] **1F1B Pipeline Parallelism** — steady-state micro-batch scheduling, activation memory bounding

### 8.7 Agentic Memory Graphs
- [x] **Dynamic Fact Extraction** — atomic entity-predicate-object knowledge extraction
- [x] **Memory Graph Consolidation** — temporal contradiction resolution and ground-truth maintenance

---

## Mental Model Checklist (Can You Answer These?)

*Detailed technical deep dives and production answers available in [`docs/mental-models-guide.md`](file:///Users/santosh/workspace/github/ai-systems-lab/docs/mental-models-guide.md).*

### API & Tokens
- [x] **Why does streaming use SSE and not WebSockets?** (Unidirectional token stream, HTTP/2 multiplexing, native proxy & connection caching)
- [x] **What is a token? Why does "ChatGPT" cost more tokens than "GPT"?** (BPE vocabulary subword frequency dictionary)
- [x] **Why is output more expensive than input?** (Compute-bound parallel matrix prefill vs memory-bandwidth-bound sequential autoregressive decoding)

### Patterns
- [x] **What is the difference between RAG and fine-tuning? When do you use each?** (Factual real-time knowledge vs stylistic/dialect/structural model behavior)
- [x] **How does function calling work at the protocol level?** (Schema injection, grammar constraints, tool execution boundary, context resubmission)
- [x] **What is the ReAct loop? Draw it on a whiteboard.** (Thought -> Action -> Observation cycle)
- [x] **How do you prevent an agent from looping forever?** (Iteration limits, call hash cycle detection, token caps, self-correction prompts)

### System Design
- [x] **How do you handle a 429 from OpenAI in a production system?** (Exponential backoff with full jitter, header tracking, LiteLLM failover routing)
- [x] **How do you know if a prompt change degraded quality?** (Golden dataset regression assertions, LLM-as-a-judge, shadow traffic A/B tests)
- [x] **How would you design an AI API gateway for 1M requests/day?** (Redis semantic embedding cache, token bucket rate limits, weighted round-robin, async trace streams)
- [x] **What would you monitor in an LLM-based production system?** (TTFT, ITL, token count / cost per tenant, guardrail block rate, refusal rate)

### Trade-offs
- [x] **When do you use GPT-4o vs GPT-4o-mini?** (Complex reasoning/code vs fast extraction/routing/guardrails)
- [x] **When is RAG better than a long context window?** (Cost reduction, sub-second TTFT latency, precision, hallucination suppression)
- [x] **When do you use an agent vs a chain vs a single prompt?** (Autonomous multi-step investigation vs fixed DAG pipeline vs single transformation)
