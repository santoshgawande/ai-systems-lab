# Claude Code Autonomous Task Queue

Edit this file freely. The runner picks up `- [ ]` lines top-to-bottom.
Status markers: `[ ]` pending · `[x]` done · `[!]` failed · `[~]` skipped

---

## AI Systems Lab — Enhancement Tasks

### Section improvements
- [ ] Add a working LangGraph lab to a new section 29-langgraph: create 01-basics with a simple node→edge→node workflow that calls Ollama, and a README explaining nodes, edges, and state schema
- [ ] Create section 30-streaming-patterns with two labs: 01-sse-basics showing raw Server-Sent Events with FastAPI, and 02-token-streaming showing how to stream LLM tokens to a browser with live word-by-word rendering
- [ ] Add a lab 03-selective-retrieval to 27-context-management: implement episodic memory using Redis (proxmox1:6379) — store conversation turns as embeddings, retrieve only the top-3 relevant past turns for each new message
- [ ] Create section 31-prompt-caching with two labs: 01-anthropic-cache using the anthropic SDK cache_control parameter, and 02-openai-cache showing how OpenAI automatic caching works and how to measure cache hit rate with the usage.prompt_tokens_details field
- [ ] Add lab 04-model-router to 05-system-design: implement a cost-aware router that classifies requests by complexity (simple/medium/complex) and routes to haiku/sonnet/opus accordingly, log model used and estimated cost per request
- [ ] Create lab 03-constrained-decoding in 19-instructor: implement Outlines-style constrained generation using a regex pattern to guarantee phone number format output, then compare with instructor Pydantic validation approach
- [ ] Add a lab 05-rag-evals to 08-evals: implement RAGAS metrics (faithfulness, answer relevancy, context precision) over the mini-rag pipeline using synthetic QA pairs generated from the corpus

### Build-your-own additions
- [ ] Create 06-build-your-own/mini-memory-agent: a chatbot that stores each conversation turn as an embedding in a local SQLite+vector table, retrieves the top-3 semantically similar past turns on each message, and includes them as context — demonstrating episodic memory without a vector DB server
- [ ] Create 06-build-your-own/mini-structured-extractor: a pipeline that takes any URL or PDF path, extracts text, then uses instructor+Claude to extract structured data (title, date, entities, summary, key_facts) as a Pydantic model — includes batch processing of multiple documents

### Deep-dive labs
- [ ] Add lab 04-vllm-inference to 25-huggingface: explain vLLM vs Ollama architecture (continuous batching, PagedAttention), show how to start a vLLM server with python -m vllm.entrypoints.openai.api_server, and benchmark throughput vs Ollama for the same model
- [ ] Create section 32-observability with two labs: 01-langsmith showing how to trace LLM calls with @traceable decorator and view traces in LangSmith, and 02-custom-tracing implementing a lightweight tracer class that logs prompt/completion/latency/cost to a local SQLite database
- [ ] Add lab 03-adversarial-robustness to 15-ai-security: implement a systematic robustness test that sends 20 adversarial variants of the same prompt (negation, paraphrase, multilingual, encoding tricks) and measures which variants cause the model to change its answer

### Polish and documentation
- [ ] Review every lab in 21-reranking through 27-context-management and add a concrete "production checklist" section to each README covering: when to use this pattern, what to monitor in production, and common failure modes
- [ ] Add a top-level QUICKSTART.md to the repo root that lists the 5 most important labs to do first for someone new to AI engineering, with estimated time and prerequisites for each

---

## How to add your own tasks

Add a line anywhere in this file:

```
- [ ] Your task description here
```

The more specific you are, the better Claude Code does:
- Bad:  `- [ ] improve the RAG section`
- Good: `- [ ] Add semantic chunking to 03-rag/02-indexing: split text on sentence boundaries using spaCy, compare chunk overlap and retrieval quality vs fixed-size chunks`
