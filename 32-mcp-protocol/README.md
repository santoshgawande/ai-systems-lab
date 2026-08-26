# 32. Model Context Protocol (MCP)

Anthropic's Model Context Protocol (MCP) has emerged as the universal standard for AI tool and data interoperability, replacing bespoke API wrappers with standard JSON-RPC 2.0 servers.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-mcp-server` | MCP JSON-RPC 2.0 Server | Handshake, capabilities, tool registration, resource handling |
| `02-mcp-client` | MCP Client & Tool Dispatch | Multi-server connections, schema translation, dynamic tool dispatch |

## Key Concepts

- **Client-Server Architecture**: The AI client (host application like Claude Code) connects to one or more MCP servers (e.g. Postgres, GitHub, Filesystem).
- **Three Core Primitives**:
  - **Tools**: Executable functions that modify external state or run computations (`tools/call`).
  - **Resources**: Read-only context data, file contents, or database schemas (`resources/read`).
  - **Prompts**: Pre-engineered prompt templates with parameters (`prompts/get`).
