# Lab 04 — MCP (Model Context Protocol) Concepts

MCP is how Claude Code extends its tool set. Understand the protocol before building servers.

## What you learn

- What MCP is: JSON-RPC 2.0 over stdio (or HTTP/SSE)
- The three MCP primitives: **tools**, **resources**, **prompts**
- How Claude Code discovers and calls MCP servers
- How to build a minimal MCP server in Python
- How to wire it into Claude Code via `~/.claude/settings.json`

## Run

```bash
pip install mcp
python mcp_demo.py
```

## MCP architecture

```
Claude Code (MCP client)
    ↕ JSON-RPC over stdio
MCP Server (your Python process)
    ↕ local function calls
Your tools (filesystem, API, database, etc.)
```

## The three primitives

| Primitive | What it is | Example |
|---|---|---|
| **Tool** | A callable function with a schema | `search_docs`, `run_query`, `send_email` |
| **Resource** | Read-only data the model can access | `file://project.txt`, `db://schema` |
| **Prompt** | A reusable prompt template | `code-review`, `summarize-pr` |

## Wire into Claude Code

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/path/to/mcp_demo.py"],
      "env": {}
    }
  }
}
```

Then in Claude Code: `/mcp` to see available tools, or just ask Claude to use them.
