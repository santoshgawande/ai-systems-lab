# Lab 07 — Token Counting & MCP Overhead

Understand how input/output tokens are counted, and how MCP tool use inflates
your token bill — then learn how to cut it.

## Labs

| File | What you learn |
|------|---------------|
| `01_tokenizer_basics.py` | How text splits into tokens; why JSON is expensive |
| `02_ollama_usage_fields.py` | Read real token counts from Ollama's API response |
| `03_mini_mcp_server.py` | A working MCP server with 3 tools (weather, calc, DB) |
| `04_mcp_token_overhead.py` | **Core lab** — plain prompt vs MCP, side-by-side |
| `05_cost_optimisation.py` | 6 concrete strategies to cut MCP token cost |

## Quick start (Docker)

```bash
# The KEY lab — MCP token overhead (no Ollama needed)
docker compose --profile lab04 up --build

# Tokenizer basics
docker compose --profile lab01 up --build

# Cost optimisation strategies
docker compose --profile lab05 up --build

# Ollama labs (pulls the model on first run — takes a few minutes)
docker compose --profile ollama up -d          # start Ollama
docker exec <ollama_container> ollama pull llama3.2
docker compose --profile ollama up lab02-ollama-usage
```

## What MCP adds to your token bill

```
PLAIN PROMPT
  system prompt           ~50 tokens
  user message            ~15 tokens
  ─────────────────────────────────
  INPUT TOTAL             ~65 tokens

WITH MCP (3 tools registered)
  system prompt           ~50 tokens
  tool definitions       ~300 tokens   ← always present, even if no tool called
  user message            ~15 tokens
  tool_call JSON          ~30 tokens   ← output turn 1 → becomes input turn 2
  tool_result JSON        ~50 tokens   ← fed back as input
  ─────────────────────────────────
  INPUT TOTAL            ~445 tokens   → 6.8x more expensive
```

## Key insight

Tool definitions are injected into **every request**, whether or not a tool is
called. Three tools with verbose descriptions = ~300 extra input tokens on
every single API call.
