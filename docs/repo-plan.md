# AI Systems Lab — Repo Plan

Separate repo (`ai-systems-lab`) for learning how production AI systems work as a senior software engineer.
Covers Claude Code, ChatGPT, Gemini internals and the engineering patterns behind them.

---

## Repo Structure

```
ai-systems-lab/
│
├── README.md
│
├── 01-llm-apis/                    # Foundation: raw API usage & internals
│   ├── openai-basics/              # ChatGPT API, tokens, context window
│   ├── claude-basics/              # Anthropic API, system prompts
│   └── gemini-basics/              # Google Gemini API
│
├── 02-core-concepts/               # Concepts used by all AI systems
│   ├── tokenization/               # How text → tokens works
│   ├── context-window/             # Context limits, chunking strategies
│   ├── streaming/                  # SSE, streaming responses
│   └── embeddings/                 # Vector embeddings, similarity search
│
├── 03-patterns/                    # Engineering patterns AI systems use
│   ├── rag/                        # Retrieval-Augmented Generation
│   ├── tool-use/                   # Function calling / tool use
│   ├── agents/                     # Agentic loops, ReAct pattern
│   ├── prompt-caching/             # Cache strategies for LLM calls
│   └── multi-turn/                 # Conversation state management
│
├── 04-system-design/               # How production AI systems are built
│   ├── rate-limiting/              # LLM API rate limits + handling
│   ├── cost-tracking/              # Token usage, cost estimation
│   ├── observability/              # Tracing LLM calls (LangSmith/Arize)
│   └── gateway/                    # Build an AI API gateway
│
├── 05-build-your-own/              # Simplified clones to learn internals
│   ├── mini-claude-code/           # CLI agent that edits files
│   ├── mini-chatgpt/               # Stateful chat with memory
│   └── mini-rag/                   # Document Q&A pipeline
│
└── learning_plan.md
```

---

## What You Learn Per Section

| Section | What You Learn |
|---|---|
| `01-llm-apis` | API auth, rate limits, model differences, pricing |
| `02-core-concepts` | Tokens, embeddings, streaming (SSE), context management |
| `03-patterns` | RAG, tool-calling, agent loops — used in all real AI products |
| `04-system-design` | How Anthropic/OpenAI build production infra around LLMs |
| `05-build-your-own` | Deepest learning — build simplified versions of real systems |

---

## Recommended Learning Order

1. `01-llm-apis` — get comfortable with raw APIs across providers
2. `02-core-concepts/streaming` — Claude Code, ChatGPT all stream responses via SSE
3. `03-patterns/tool-use` — function calling is the core of how Claude Code works
4. `03-patterns/agents` — agentic loop (observe → think → act) is the heart of Claude Code
5. `05-build-your-own/mini-claude-code` — build a simplified version to solidify everything

---

## Key Concepts Behind Each AI System

### Claude Code
- Agentic loop with tool use (read file, write file, run bash)
- System prompt with extensive instructions
- Streaming responses (SSE)
- Context window management across long sessions

### ChatGPT
- Multi-turn conversation state (messages array)
- Function calling / tool use
- Plugin/GPT architecture
- RLHF-trained models

### Gemini
- Multimodal inputs (text, image, video, audio)
- Long context window (up to 1M tokens)
- Grounding with Google Search
- Similar tool-use API pattern to OpenAI

---

## Related Repo
- `system-design-labs` — HTTP, databases, caching, distributed systems fundamentals
