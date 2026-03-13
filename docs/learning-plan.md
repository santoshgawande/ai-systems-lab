# AI Systems Learning Plan — Senior Software Engineer

> Goal: Understand how production AI systems (Claude Code, ChatGPT, Gemini, Copilot, etc.)
> are designed, built, and operated — from API to infra to evaluation.

---

## How to Use This Plan

- Work through phases **in order** — each builds on the last
- Each topic has a **check** — the thing you should be able to do/explain after learning it
- Mark topics `[x]` as you complete them
- Build something in every phase — reading alone won't make it stick

---

## Phase 1 — LLM API Foundations
> Get fluent with raw APIs before learning patterns built on top of them.

### 1.1 API Basics
- [ ] **OpenAI API** — completions, chat format, models, auth, pricing
  - Check: call GPT-4o, inspect the request/response JSON, understand `messages` array
- [ ] **Anthropic API** — system prompts, `messages` format, model tiers
  - Check: call claude-sonnet-4-6, use a system prompt, understand how it differs from OpenAI
- [ ] **Gemini API** — multimodal inputs, long context (1M tokens), grounding
  - Check: send an image + text prompt, observe the response

### 1.2 Token Mechanics
- [ ] **Tokenization** — how text maps to tokens (BPE, tiktoken, sentencepiece)
  - Check: tokenize a sentence, count tokens manually, explain why "ChatGPT" is 2 tokens
- [ ] **Context window** — prompt tokens + completion tokens = max window
  - Check: hit a context limit, handle the error, implement chunking
- [ ] **Pricing model** — input vs output token costs, why output is more expensive
  - Check: estimate cost of 1000 API calls for a given use case

### 1.3 Streaming
- [ ] **SSE (Server-Sent Events)** — how LLMs stream tokens to client
  - Check: implement a streaming client in Python or Node.js, render tokens as they arrive
- [ ] **Backpressure** — what happens if the client can't keep up
  - Check: explain how SSE handles slow consumers vs WebSockets

---

## Phase 2 — Core Engineering Concepts
> The primitives all AI systems are built from.

### 2.1 Embeddings
- [ ] **What embeddings are** — dense vector representation of text
  - Check: embed 10 sentences, compute cosine similarity, find nearest neighbors
- [ ] **Embedding models** — text-embedding-3-small, voyage-3, Gemini embeddings
  - Check: compare embedding dimensions and quality trade-offs
- [ ] **Vector stores** — pgvector, Pinecone, Weaviate, Chroma, Qdrant
  - Check: store 1000 embeddings in pgvector, run a similarity query

### 2.2 Prompt Engineering (Engineering, Not Magic)
- [ ] **Few-shot prompting** — examples in context improve outputs
- [ ] **Chain-of-thought** — step-by-step reasoning improves accuracy
- [ ] **Structured output** — JSON mode, response_format, constrained decoding
  - Check: make an LLM always return valid JSON, handle parse failures
- [ ] **System prompt design** — role, constraints, output format, examples
  - Check: write a system prompt that reliably produces consistent behavior

### 2.3 Context Management
- [ ] **Sliding window** — trim old messages to fit context
- [ ] **Summarization** — compress old turns into a summary
- [ ] **Selective retrieval** — only include relevant history
  - Check: implement a multi-turn chat that stays within a 4k token budget

---

## Phase 3 — Patterns (The Core of All AI Products)
> Every real AI product is one or more of these patterns composed together.

### 3.1 RAG — Retrieval-Augmented Generation
- [ ] **Chunking strategies** — fixed size, sentence, semantic, recursive
- [ ] **Indexing pipeline** — ingest → chunk → embed → store
- [ ] **Retrieval** — similarity search, MMR, hybrid (BM25 + vector)
- [ ] **Generation** — stuff chunks into prompt, instruct model to answer from context
- [ ] **Re-ranking** — cohere rerank, cross-encoder to improve top-k
  - Check: build a document Q&A pipeline over a PDF from scratch

### 3.2 Tool Use / Function Calling
- [ ] **OpenAI function calling** — define tools as JSON schema, parse model's tool calls
- [ ] **Anthropic tool use** — same pattern, different API shape
- [ ] **Parallel tool calls** — model calls multiple tools at once
- [ ] **Tool result handling** — feed results back into context
  - Check: build an agent that can call a weather API + calculator + search engine

### 3.3 Agentic Loops
- [ ] **ReAct pattern** — Reason + Act: think → tool call → observe → repeat
- [ ] **Plan-and-Execute** — plan all steps upfront, execute each
- [ ] **Agentic loop termination** — stop conditions, max iterations, error handling
- [ ] **Human-in-the-loop** — when to pause and ask for confirmation
  - Check: build a CLI agent that can read/write files and run shell commands

### 3.4 Memory Systems
- [ ] **In-context memory** — conversation history in the prompt (ephemeral)
- [ ] **External memory** — vector DB, key-value store (persistent)
- [ ] **Episodic memory** — store and retrieve past interactions
- [ ] **Semantic memory** — facts and knowledge the agent knows about a user
  - Check: build a chatbot that remembers user preferences across sessions

### 3.5 Prompt Caching
- [ ] **Anthropic prompt caching** — cache prefix tokens, save cost on repeated system prompts
- [ ] **OpenAI prompt caching** — automatic caching for long prompts
  - Check: measure cache hit rate and cost savings on a real workload

### 3.6 Multi-Agent Systems
- [ ] **Orchestrator + subagent pattern** — one agent delegates to specialized agents
- [ ] **Message-passing** — agents communicate via structured messages
- [ ] **Shared context / blackboard** — agents share state via a central store
- [ ] **Agent-as-tool** — one agent calls another as if it were a function
  - Check: build a 2-agent system (planner + executor) for a coding task

---

## Phase 4 — System Design for AI
> How to build production-grade systems around LLMs.

### 4.1 Reliability
- [ ] **Rate limit handling** — exponential backoff, jitter, queue with token bucket
- [ ] **Retry strategy** — which errors to retry (429, 503) vs not (400, 401)
- [ ] **Fallback routing** — if primary model fails, fall back to secondary
- [ ] **Timeout handling** — streaming timeouts, partial response recovery
  - Check: implement a resilient LLM client with retry + fallback

### 4.2 Cost Optimization
- [ ] **Token budgeting** — limit output tokens, compress prompts
- [ ] **Model routing** — use cheap model (Haiku) for simple tasks, expensive (Opus) for hard ones
- [ ] **Caching** — cache identical or near-identical prompts
- [ ] **Batching** — batch small requests to reduce overhead
  - Check: reduce per-call cost by 50% on a sample workload using these techniques

### 4.3 Observability
- [ ] **LLM tracing** — log every prompt + completion (LangSmith, Arize, Braintrust)
- [ ] **Token usage metrics** — track input/output tokens per request, per user, per day
- [ ] **Latency tracking** — time-to-first-token (TTFT), total latency, p50/p95/p99
- [ ] **Error tracking** — classify errors (model, rate limit, timeout, parse)
  - Check: instrument an LLM app so you can debug any failure from logs alone

### 4.4 AI Gateway / LLM Proxy
- [ ] **What a gateway does** — auth, routing, rate limiting, logging, caching in one layer
- [ ] **Open source options** — LiteLLM, Portkey, OpenRouter
- [ ] **Build your own** — reverse proxy that adds auth + logging + fallback
  - Check: route requests to 3 different LLM providers through a single endpoint

### 4.5 Evaluation (Evals)
- [ ] **Why evals matter** — "prompt changes break things silently"
- [ ] **Types of evals** — unit evals, model-graded, human eval, A/B test
- [ ] **Eval frameworks** — Braintrust, PromptFoo, RAGAS (for RAG)
- [ ] **Regression testing** — run evals on every prompt change in CI
  - Check: write 20 evals for a RAG pipeline, run them in CI

### 4.6 Safety & Guardrails
- [ ] **Input guardrails** — detect prompt injection, jailbreaks, PII
- [ ] **Output guardrails** — validate JSON, detect hallucinations, filter harmful content
- [ ] **Constitutional AI** — self-critique + revision
- [ ] **Moderations API** — OpenAI moderation, Anthropic's usage policies
  - Check: add an input + output guard to an existing agent

---

## Phase 5 — Advanced Topics
> For going deep beyond standard patterns.

### 5.1 Fine-Tuning vs Prompting
- [ ] **When to fine-tune** — consistent style/format, domain-specific knowledge, cost reduction
- [ ] **When to prompt** — flexibility, fast iteration, small datasets
- [ ] **Fine-tuning OpenAI** — JSONL dataset, training job, evaluation
- [ ] **LoRA / QLoRA** — parameter-efficient fine-tuning for open-source models
  - Check: fine-tune a small model for a classification task, compare vs few-shot

### 5.2 Multimodal Systems
- [ ] **Vision** — GPT-4o vision, Claude's vision, Gemini — image + text inputs
- [ ] **Audio** — Whisper (STT), TTS, voice agents
- [ ] **Document parsing** — PDF, tables, structured extraction
  - Check: build a pipeline that extracts data from a scanned invoice image

### 5.3 Structured Generation
- [ ] **JSON mode** — guaranteed valid JSON output
- [ ] **Pydantic + instructor** — parse LLM output directly into typed objects
- [ ] **Constrained decoding** — Outlines, guidance — token-level control
  - Check: extract structured data from 100 free-text records reliably

### 5.4 Long Context Strategies
- [ ] **Lost-in-the-middle problem** — models miss info in middle of long contexts
- [ ] **Positional encoding limits** — why context degrades beyond training length
- [ ] **Chunking + stitching** — process long docs in sections, merge results
  - Check: summarize a 200-page document accurately without losing key details

### 5.5 Open-Source Models
- [ ] **Ollama / vLLM** — run Llama 3, Mistral, Qwen locally
- [ ] **HuggingFace Transformers** — load, run, and understand open models
- [ ] **Quantization** — GGUF, GPTQ, AWQ — run large models on consumer hardware
  - Check: run a 7B model locally, compare quality and speed vs GPT-4o-mini

---

## Phase 6 — Build Your Own (Internalize by Building)
> The highest-leverage learning. Build simplified versions of real AI systems.

| Project | What It Teaches |
|---|---|
| `mini-claude-code` | Agentic loop, tool use, file I/O, streaming CLI |
| `mini-chatgpt` | Multi-turn state, memory, streaming UI |
| `mini-rag` | Full RAG pipeline: ingest → chunk → embed → retrieve → generate |
| `mini-copilot` | Code completion, context injection, IDE extension basics |
| `mini-eval-framework` | How to test AI systems like software |
| `mini-ai-gateway` | LLM proxy with auth, routing, logging, fallback |

---

## Mental Model Checklist (Can You Answer These?)

### API & Tokens
- [ ] Why does streaming use SSE and not WebSockets?
- [ ] What is a token? Why does "ChatGPT" cost more tokens than "GPT"?
- [ ] Why is output more expensive than input?

### Patterns
- [ ] What is the difference between RAG and fine-tuning? When do you use each?
- [ ] How does function calling work at the protocol level?
- [ ] What is the ReAct loop? Draw it on a whiteboard.
- [ ] How do you prevent an agent from looping forever?

### System Design
- [ ] How do you handle a 429 from OpenAI in a production system?
- [ ] How do you know if a prompt change degraded quality?
- [ ] How would you design an AI API gateway for 1M requests/day?
- [ ] What would you monitor in an LLM-based production system?

### Trade-offs
- [ ] When do you use GPT-4o vs GPT-4o-mini?
- [ ] When is RAG better than a long context window?
- [ ] When do you use an agent vs a chain vs a single prompt?

---

## Resources

### Books
- _Building LLMs for Production_ — Towards AI
- _AI Engineering_ — Chip Huyen (2025)

### Courses
- DeepLearning.AI short courses (LangChain, RAG, Agents, Evals)
- Fast.ai — for understanding model internals

### Papers to Read
- Attention Is All You Need (Transformer architecture)
- ReAct: Synergizing Reasoning and Acting in Language Models
- REALM / RAG original paper
- Constitutional AI (Anthropic)

### Tools to Know
- LiteLLM — multi-provider LLM client
- LangSmith / Braintrust — eval + tracing
- Instructor — structured LLM outputs
- RAGAS — RAG evaluation
- vLLM / Ollama — self-hosted inference
- Portkey / OpenRouter — AI gateway

---

## Completion Criteria

You're done with a topic when you can:
1. **Explain it** to another engineer in 2 minutes
2. **Build it** from scratch without tutorials
3. **Reason about trade-offs** — when to use it vs alternatives
