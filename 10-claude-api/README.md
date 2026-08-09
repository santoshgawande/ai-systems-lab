# 10 — Claude API (Anthropic)

Anthropic-specific features: the Messages API format, prompt caching, extended thinking, tool use, and MCP.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get an API key: https://console.anthropic.com/

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

> Labs fall back to Ollama if `ANTHROPIC_API_KEY` is not set — conceptually equivalent but without Claude-specific features (prompt caching, extended thinking).

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-basics/` | Messages API format, system prompts, vision, tool use — how it differs from OpenAI | `python claude_basics.py` |
| `02-prompt-caching/` | Cache large system prompts — measure 90% cost reduction on repeated calls | `python caching.py` |
| `03-extended-thinking/` | Claude's "think before answering" — budget tokens for reasoning | `python thinking.py` |
| `04-mcp-concepts/` | How MCP works: protocol, server structure, tool registration, Claude Code integration | `python mcp_demo.py` |

## Claude vs OpenAI API differences

| Feature | OpenAI | Anthropic |
|---|---|---|
| System prompt | `messages[0].role = "system"` | Separate `system` parameter |
| Token counting | `usage.prompt_tokens` | `usage.input_tokens` |
| Tool use | `tools` + `tool_choice` | `tools` + `tool_choice` (same concept) |
| Prompt caching | Automatic (>1024 tokens) | Explicit `cache_control` marker |
| Extended thinking | o1/o3 via `reasoning_effort` | `thinking` parameter with budget |
| Vision | `content` array with image_url | `content` array with `image` type |

## Why prompt caching matters for Claude Code

Claude Code attaches a large system prompt (~50k tokens) on every request.
Without caching: 50k × $3/1M = $0.15 per request just for the system prompt.
With caching: 5k write + 45k read × $0.30/1M = $0.017 per request.
**~88% cheaper** — this is why Claude Code is economically viable at scale.
