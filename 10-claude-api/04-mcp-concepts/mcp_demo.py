#!/usr/bin/env python3
"""
Minimal MCP server demonstrating all three primitives: tools, resources, prompts.
Install: pip install mcp
Wire into Claude Code via ~/.claude/settings.json (see README).
"""
import asyncio
import json
import math
import datetime
import httpx
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

server = Server("lab-mcp-server")

# ─── TOOLS: callable functions the model can invoke ──────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculator",
            description="Evaluate a mathematical expression safely.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. 'sqrt(144)', '2 ** 10'"
                    }
                },
                "required": ["expression"]
            }
        ),
        types.Tool(
            name="get_datetime",
            description="Get the current date and time.",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="fetch_url",
            description="Fetch the content of a URL (first 2000 chars).",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "calculator":
        try:
            safe = {k: v for k, v in vars(math).items() if not k.startswith("_")}
            result = eval(arguments["expression"], {"__builtins__": {}}, safe)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    elif name == "get_datetime":
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        return [types.TextContent(type="text", text=now)]

    elif name == "fetch_url":
        try:
            r = httpx.get(arguments["url"], timeout=10, follow_redirects=True)
            return [types.TextContent(type="text", text=r.text[:2000])]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error fetching URL: {e}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ─── RESOURCES: read-only data the model can access ──────────────────────────

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="memo://system-info",
            name="System Information",
            description="Current system datetime and Python version",
            mimeType="text/plain"
        ),
        types.Resource(
            uri="memo://math-constants",
            name="Math Constants",
            description="Common mathematical constants",
            mimeType="application/json"
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    import sys
    if uri == "memo://system-info":
        return f"DateTime: {datetime.datetime.now()}\nPython: {sys.version}"
    elif uri == "memo://math-constants":
        return json.dumps({
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "golden_ratio": (1 + math.sqrt(5)) / 2
        }, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


# ─── PROMPTS: reusable prompt templates ──────────────────────────────────────

@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="code-review",
            description="Review code for bugs, security issues, and improvements.",
            arguments=[
                types.PromptArgument(name="code", description="The code to review", required=True),
                types.PromptArgument(name="language", description="Programming language", required=False),
            ]
        ),
        types.Prompt(
            name="explain-error",
            description="Explain an error message and suggest fixes.",
            arguments=[
                types.PromptArgument(name="error", description="The error message", required=True),
            ]
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    args = arguments or {}
    if name == "code-review":
        lang = args.get("language", "unknown")
        code = args.get("code", "")
        return types.GetPromptResult(
            description="Code review prompt",
            messages=[types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Review this {lang} code for bugs, security issues, and improvements:\n\n```{lang}\n{code}\n```\n\nProvide specific, actionable feedback."
                )
            )]
        )
    elif name == "explain-error":
        error = args.get("error", "")
        return types.GetPromptResult(
            description="Error explanation prompt",
            messages=[types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Explain this error and suggest how to fix it:\n\n```\n{error}\n```"
                )
            )]
        )
    raise ValueError(f"Unknown prompt: {name}")


# ─── Entry point ──────────────────────────────────────────────────────────────

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="lab-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
