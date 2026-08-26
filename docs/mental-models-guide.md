# Senior AI Systems Engineering — Mental Models & Deep Dives

This document provides rigorous, production-tested answers to the core mental model checklist questions in [learning-plan.md](file:///Users/santosh/workspace/github/ai-systems-lab/docs/learning-plan.md).

---

## 1. API & Token Mechanics

### Q1: Why does streaming use SSE (Server-Sent Events) and not WebSockets?
* **Unidirectional Nature**: LLM token generation is strictly one-way (server to client). The client sends one HTTP POST request with the prompt, and the server emits tokens over time.
* **HTTP/2 & HTTP/3 Multiplexing**: SSE operates over standard HTTP/HTTPS (`text/event-stream`), natively inheriting browser HTTP connection pooling, TLS renegotiation, edge proxy caching, and firewall traversal without custom WS upgrade handshakes.
* **Simpler Infrastructure**: WebSockets require stateful sticky sessions, custom load balancer configuration (e.g. Envoy/NGINX connection persistence), and bi-directional ping-pong heartbeats. SSE auto-reconnects with the `Last-Event-ID` header.

### Q2: What is a token? Why does "ChatGPT" cost more tokens than "GPT"?
* **Byte-Pair Encoding (BPE)**: Tokenizers break text into common subword chunks based on training corpus frequency.
* **Subword Frequency**: Common words/acronyms like `"GPT"` frequently appear as a single dedicated token in vocabulary dictionaries (e.g. token ID `397` in `cl100k_base`).
* **Compound Words**: `"ChatGPT"` was coined later and is typically split into 2 tokens: `["Chat", "GPT"]` (or `["Ch", "at", "GPT"]` in older tokenizers like `p50k_base`).

### Q3: Why is output token generation more expensive than input token processing?
* **Parallel Prefill vs. Sequential Autoregressive Decoding**:
  * **Input Prefill**: The model processes all $N$ input tokens simultaneously in a single compute-bound GEMM (Matrix Multiplication) pass using GPU tensor cores at near 100% compute utilization.
  * **Output Decoding**: Output tokens MUST be generated sequentially one-by-one. Each token generation step requires loading the ENTIRE model weight parameters (e.g. 140 GB for 70B FP16) from HBM into SRAM just to emit a single token, making decoding memory-bandwidth bound.

---

## 2. Architectural Patterns

### Q4: What is the difference between RAG and Fine-Tuning? When do you use each?
| Dimension | Retrieval-Augmented Generation (RAG) | Fine-Tuning (SFT / LoRA / DPO) |
|---|---|---|
| **Primary Purpose** | Supplying factual knowledge, proprietary live documents, and citations. | Teaching style, tone, structured output formats, or domain-specific reasoning habits. |
| **Knowledge Recency** | Real-time (instant updates by updating vector/graph index). | Static snapshot as of training time; requires retraining on data changes. |
| **Hallucination Risk** | Low (grounded in retrieved context with exact source attribution). | High (model can hallucinate unsupported "facts" embedded in weights). |
| **Data Requirements** | Unstructured text documents (PDFs, Markdown, DB rows). | Curated pairs of $(prompt, target\_completion)$ or $(prompt, chosen, rejected)$. |
| **Best Used For** | Enterprise search, customer support docs, codebases, knowledge bases. | Adhering to complex custom JSON schemas, specialized SQL dialects, coding style. |

### Q5: How does function calling work at the protocol level?
1. **Schema Injection**: Client sends a list of JSON Schema tool definitions in the API payload (`tools=[{"name": "...", "parameters": {...}}]`).
2. **Grammar / Special Token Constraint**: The model generates a special delimiter token (e.g. `<tool_call>` or JSON mode grammar constraint) containing the function name and argument key-values.
3. **Execution Barrier**: The client/runtime stops generation, parses the arguments, executes the local Python/HTTP function, and formats the output as a `role: "tool"` message.
4. **Context Loop**: The client resends the full conversation history including the tool output back to the LLM to generate the final user-facing synthesized answer.

### Q6: What is the ReAct loop?
```
           +--------------------------+
           |       User Prompt        |
           +--------------------------+
                        |
                        v
        +---> [ Thought: Reason ]
        |               |
        |               v
        |      [ Action: Tool Call ]
        |               |
        |               v
        |    [ Observation: Result ]
        |               |
        +------- (Done? No)
                        | (Done? Yes)
                        v
              [ Final Answer ]
```
1. **Reason**: LLM generates internal reasoning step ("I need to find the current stock price of AAPL").
2. **Act**: LLM emits structured tool invocation (`get_stock_quote(symbol='AAPL')`).
3. **Observe**: Runtime executes the tool and injects the raw result into context.
4. **Iterate**: The loop repeats until the model generates the termination condition / final answer.

### Q7: How do you prevent an agent from looping forever?
1. **Hard Iteration Budget**: Enforce a strict `max_iterations = 10` counter in the orchestrator loop.
2. **Cycle / Duplicate Detection**: Maintain a sliding window hash of recent `(tool_name, arguments)` calls; abort or inject a prompt warning if the identical tool call occurs 2+ times consecutively.
3. **Token & Timeout Hard-Caps**: Enforce `max_tokens` per step and an overall task timeout (e.g. `timeout=60s`).
4. **Self-Correction Interventions**: If an error repeats, inject a system message: *"You have called `search()` 3 times with no new results. Provide your best answer now."*

---

## 3. Production System Design

### Q8: How do you handle HTTP 429 (Rate Limits) in production?
1. **Exponential Backoff with Full Jitter**:
   $$\text{sleep} = \min(t_{\max}, t_{\text{base}} \cdot 2^{\text{attempt}}) \times \text{Uniform}(0.5, 1.5)$$
2. **Header Adherence**: Check `Retry-After` or `x-ratelimit-reset-requests` headers in the 429 response and sleep for the exact duration specified.
3. **Tiered Fallback Routing**: Immediately failover from primary provider (e.g. Azure OpenAI) to secondary provider (e.g. Direct OpenAI or AWS Bedrock) via LiteLLM.
4. **Client-Side Token Bucket / Leaky Bucket**: Rate-limit requests upstream at the API gateway before sending them to the provider.

### Q9: How do you know if a prompt change degraded quality?
1. **Golden Dataset Regression Suite**: Run automated unit assertions and LLM-as-a-Judge evals on a curated dataset of 100+ representative test cases in CI/CD before deploying.
2. **Deterministic Metric Tracking**: Measure JSON parsing pass rate, exact keyword recall, and schema validation error rates.
3. **Shadow Traffic / A/B Testing**: Send 5% of live traffic to the candidate prompt, comparing user thumbs-up/thumbs-down ratings and session completion rates against the control prompt.

### Q10: How do you design an AI API Gateway for 1,000,000 requests/day?
* **Architecture**: Stateless Go or Rust reverse proxy (Envoy / LiteLLM Proxy) behind Cloudflare / AWS ALB.
* **Semantic Caching**: Redis cluster storing embedding vectors of incoming prompts; returns cached responses for cosine similarity $>0.98$ (saving 20–30% API cost).
* **Provider Load Balancing**: Weighted round-robin across multiple provider accounts and regions with automatic circuit breakers on 5xx/429 errors.
* **Token Rate Limiting**: Distributed Redis Token Bucket limiting RPM (Requests Per Minute) and TPM (Tokens Per Minute) per API key.
* **Async Logging**: Fire-and-forget Kafka/SQS event stream pushing prompt-completion traces to ClickHouse/Langfuse.

### Q11: What do you monitor in an LLM production system?
* **Latency**: Time-to-First-Token (TTFT), Inter-Token Latency (ITL), and Total End-to-End Duration (p50, p95, p99).
* **Token Usage & Costs**: Prompt tokens, completion tokens, and dollar cost per tenant/feature.
* **Error Classifications**: 429 Rate Limits, 503 Provider Outages, JSON schema parse errors, and tool execution failures.
* **Quality & Safety Signals**: Guardrail block rate, refusal rate, LLM-as-a-judge faithfulness scores, and user feedback metrics.

---

## 4. Architectural Trade-Offs

### Q12: When do you use GPT-4o / Claude 3.5 Sonnet vs GPT-4o-mini / Claude 3.5 Haiku?
* **Use Flagship (GPT-4o / Sonnet / Pro)** for: Multi-step reasoning, complex code generation, ambiguous instructions, mathematical logic, and critical evaluation judges.
* **Use Fast/Cheap (4o-mini / Haiku / Flash)** for: Classification, extraction, query reformulation, summarization of simple docs, embedding rerank candidates, and input guardrail filtering.

### Q13: When is RAG better than a 1M+ token context window?
* **Cost**: Stuffing 1M tokens on every turn costs $\$3.00+$ per query vs $\$0.001$ for 5 retrieved RAG chunks.
* **Latency (TTFT)**: Processing 1M tokens takes 15–30 seconds of prefill time; RAG retrieves in $<50\text{ms}$ and answers in $<1\text{s}$.
* **Lost-in-the-Middle & Precision**: Models suffer attention dilution and higher hallucination rates across giant context windows; targeted retrieval isolates the exact relevant paragraphs.

### Q14: When do you use an Agent vs a Chain vs a Single Prompt?
* **Single Prompt**: Direct single-turn transformations (e.g. grammar correction, translation, sentiment classification).
* **Deterministic Chain / DAG**: Fixed multi-step pipelines where Step B always follows Step A (e.g. `Retrieve Docs -> Extract Entities -> Format Markdown`).
* **Autonomous Agent**: Open-ended tasks where the number of steps, choice of tools, and investigation path cannot be known in advance (e.g. debugging a bug in a multi-file repo, conducting deep research across web sources).
