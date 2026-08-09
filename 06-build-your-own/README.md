# 06 — Build Your Own

Build simplified versions of real AI systems from scratch. The highest-leverage section — you can't fake understanding when you have to build it yourself.

## Projects

| Dir | What you build | What it teaches |
|---|---|---|
| `mini-claude-code/` | CLI agent that reads/writes files and runs shell commands | Agentic loop, tool dispatch, streaming output |
| `mini-rag/` | Document Q&A over a local directory | Full RAG pipeline: ingest → chunk → embed → retrieve → generate |
| `mini-chatgpt/` | Streaming multi-turn chat with persistent memory | Context window management, session state, streaming UI |
| `mini-copilot/` | Fill-in-the-middle code completion | FIM prompting, context injection, prefix/suffix pattern |
| `mini-eval-framework/` | CI-ready LLM test suite | Eval design, LLM-as-judge, severity scoring, CI integration |
| `mini-ai-gateway/` | LLM proxy with auth, routing, logging, fallback | Production gateway patterns, FastAPI middleware, multi-provider |

## Run each project

```bash
# mini-claude-code: give it a task
cd mini-claude-code
python agent.py "list all python files in /tmp and count how many there are"
python agent.py "write a hello world script to /tmp/hello.py and run it"

# mini-rag: ingest docs, then query
cd mini-rag
python app.py ingest ./docs
python app.py query "what is RAG?"

# mini-chatgpt: interactive streaming chat
cd mini-chatgpt
python chat.py

# mini-copilot: interactive code completion REPL
cd mini-copilot
python copilot.py
python copilot.py --demo        # non-interactive batch demo

# mini-eval-framework: run evals, exit 1 on failures
cd mini-eval-framework
python eval_framework.py
python eval_framework.py --ci   # for CI pipelines
python eval_framework.py --suite support --severity critical

# mini-ai-gateway: start proxy server
cd mini-ai-gateway
pip install -r requirements.txt
python gateway.py
# In another terminal: curl http://localhost:8000/v1/chat/completions ...
```

## Why build your own?

Reading docs → ~10% retention.
Using a framework → ~30% retention.
Building from scratch → ~90% retention.

When you build these you're forced to understand:
- Exactly what happens at each step and why
- Why things fail and how to fix them
- The real trade-offs that framework abstractions hide
