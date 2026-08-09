#!/usr/bin/env python3
"""
Homelab MCP server: tools for Ollama, pgvector, and system info.
Wire into Claude Code via ~/.claude/settings.json mcpServers.
Install: pip install mcp httpx psycopg2-binary
"""
import asyncio
import json
import os
import datetime
import httpx
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

server = Server("homelab-tools")

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
PG_HOST = os.environ.get("PG_HOST", "192.168.0.111")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DB = os.environ.get("PG_DB", "postgres")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "postgres")


# ─── TOOLS ───────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="ollama_list",
            description="List all Ollama models available on the homelab Mac Studio.",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="ollama_generate",
            description="Generate a completion using a local Ollama model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name, e.g. 'llama3.2', 'phi4'"},
                    "prompt": {"type": "string", "description": "The prompt to send"},
                    "system": {"type": "string", "description": "Optional system prompt"}
                },
                "required": ["model", "prompt"]
            }
        ),
        types.Tool(
            name="pg_query",
            description="Run a read-only SQL SELECT query against PostgreSQL on proxmox1.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT query (no DML allowed)"},
                },
                "required": ["sql"]
            }
        ),
        types.Tool(
            name="pg_tables",
            description="List all tables in the PostgreSQL database on proxmox1.",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="embed_text",
            description="Embed text using Ollama nomic-embed-text. Returns a 768-dim vector.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to embed"}
                },
                "required": ["text"]
            }
        ),
        types.Tool(
            name="homelab_status",
            description="Check connectivity to homelab services (Ollama, PostgreSQL, Qdrant).",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    async def text(s: str) -> list[types.TextContent]:
        return [types.TextContent(type="text", text=str(s))]

    if name == "ollama_list":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = r.json().get("models", [])
            lines = [f"{m['name']} ({m.get('size', 0) // 1e9:.1f}GB)" for m in models]
            return await text("\n".join(lines) if lines else "No models found.")
        except Exception as e:
            return await text(f"Error: {e}")

    elif name == "ollama_generate":
        try:
            payload = {
                "model": arguments["model"],
                "prompt": arguments["prompt"],
                "stream": False
            }
            if arguments.get("system"):
                payload["system"] = arguments["system"]
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            return await text(r.json().get("response", "No response"))
        except Exception as e:
            return await text(f"Error: {e}")

    elif name == "pg_query":
        sql = arguments["sql"].strip()
        if not sql.upper().startswith("SELECT"):
            return await text("Error: only SELECT queries are allowed.")
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASS, connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(100)   # limit to 100 rows
            cols = [desc[0] for desc in cur.description]
            conn.close()
            if not rows:
                return await text("No rows returned.")
            lines = ["\t".join(cols)]
            for row in rows:
                lines.append("\t".join(str(v) for v in row))
            return await text("\n".join(lines))
        except ImportError:
            return await text("psycopg2 not installed: pip install psycopg2-binary")
        except Exception as e:
            return await text(f"Error: {e}")

    elif name == "pg_tables":
        sql = "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
        return await call_tool("pg_query", {"sql": sql})

    elif name == "embed_text":
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": arguments["text"]}
                )
            emb = r.json().get("embedding", [])
            return await text(f"Vector dim={len(emb)}, first 5: {emb[:5]}")
        except Exception as e:
            return await text(f"Error: {e}")

    elif name == "homelab_status":
        status = {}

        async def ping(name: str, url: str) -> str:
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    r = await client.get(url)
                return "OK" if r.status_code < 400 else f"HTTP {r.status_code}"
            except Exception as e:
                return f"DOWN ({type(e).__name__})"

        results = await asyncio.gather(
            ping("Ollama", f"{OLLAMA_BASE}/api/tags"),
            ping("Qdrant", f"http://192.168.0.112:6333/collections"),
        )
        status["Ollama (localhost:11434)"] = results[0]
        status["Qdrant (proxmox2:6333)"] = results[1]

        try:
            import psycopg2
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASS, connect_timeout=3
            )
            conn.close()
            status[f"PostgreSQL (proxmox1:{PG_PORT})"] = "OK"
        except Exception as e:
            status[f"PostgreSQL (proxmox1:{PG_PORT})"] = f"DOWN ({e})"

        lines = [f"{svc}: {state}" for svc, state in status.items()]
        return await text("\n".join(lines))

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ─── RESOURCES ───────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="homelab://config",
            name="Homelab Configuration",
            description="Current homelab service endpoints and connection details",
            mimeType="application/json"
        ),
        types.Resource(
            uri="homelab://datetime",
            name="Current Date/Time",
            description="Current UTC datetime",
            mimeType="text/plain"
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "homelab://config":
        return json.dumps({
            "ollama": {"base_url": OLLAMA_BASE, "embed_model": "nomic-embed-text"},
            "postgresql": {"host": PG_HOST, "port": PG_PORT, "db": PG_DB},
            "qdrant": {"host": "192.168.0.112", "port": 6333},
        }, indent=2)
    elif uri == "homelab://datetime":
        return datetime.datetime.utcnow().isoformat() + "Z"
    raise ValueError(f"Unknown resource: {uri}")


# ─── PROMPTS ─────────────────────────────────────────────────────────────────

@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="sql-review",
            description="Review a SQL query for performance and security issues.",
            arguments=[
                types.PromptArgument(name="sql", description="The SQL query to review", required=True),
            ]
        ),
        types.Prompt(
            name="rag-query",
            description="Build a RAG query prompt using retrieved context chunks.",
            arguments=[
                types.PromptArgument(name="question", description="User question", required=True),
                types.PromptArgument(name="context", description="Retrieved chunks", required=True),
            ]
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    args = arguments or {}
    if name == "sql-review":
        sql = args.get("sql", "")
        return types.GetPromptResult(
            description="SQL review prompt",
            messages=[types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Review this SQL query for:\n1. SQL injection vulnerabilities\n2. Missing indexes\n3. N+1 query patterns\n4. Performance issues\n\n```sql\n{sql}\n```\n\nBe specific — reference line numbers and column names."
                )
            )]
        )
    elif name == "rag-query":
        question = args.get("question", "")
        context = args.get("context", "")
        return types.GetPromptResult(
            description="RAG generation prompt",
            messages=[types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Answer the question using ONLY the context below. If the answer is not in the context, say 'I don't have information about that.'\n\nContext:\n{context}\n\nQuestion: {question}"
                )
            )]
        )
    raise ValueError(f"Unknown prompt: {name}")


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="homelab-tools",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
