# Lab 03 — MCP Server Development

Build a real MCP server from scratch and wire it into Claude Code.

## What you learn

- MCP server architecture: JSON-RPC 2.0 over stdio or HTTP/SSE
- Tools, Resources, and Prompts — all three primitives in one server
- How to test your server before wiring it into Claude Code
- Wire into Claude Code via settings.json `mcpServers`
- Debugging: `claude mcp list`, MCP inspector

## Run

```bash
pip install mcp httpx
python mcp_server.py   # starts the server (Claude Code connects to it)
```

## Wire into Claude Code

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "homelab-tools": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "OLLAMA_BASE": "http://localhost:11434",
        "PG_HOST": "192.168.0.111"
      }
    }
  }
}
```

Then in Claude Code: just ask Claude to use your tools — no slash command needed.

## Test without Claude Code

```bash
# List tools
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python mcp_server.py

# Call a tool
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ollama_list","arguments":{}},"id":2}' | python mcp_server.py
```

## MCP primitives

| Primitive | Defined by | Used for |
|---|---|---|
| Tool | `@server.list_tools` + `@server.call_tool` | Functions Claude can invoke |
| Resource | `@server.list_resources` + `@server.read_resource` | Data Claude can read |
| Prompt | `@server.list_prompts` + `@server.get_prompt` | Reusable prompt templates |

## Debugging

```bash
# Check which MCP servers Claude Code sees
claude mcp list

# Add a server from CLI
claude mcp add my-server python /path/to/server.py

# Remove a server
claude mcp remove my-server

# MCP inspector (interactive testing)
npx @modelcontextprotocol/inspector python /path/to/server.py
```
