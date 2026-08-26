from __future__ import annotations
"""
Model Context Protocol (MCP) Client & LLM Orchestrator.

The MCP Client connects an LLM agent (e.g. Claude Code or Gemini Agent) to
remote MCP servers:
1. Connects and executes the `initialize` handshake.
2. Discovers remote tools (`tools/list`).
3. Converts MCP JSON schemas to LLM provider function schemas (OpenAI / Anthropic).
4. Routes LLM tool invocations to the respective MCP Server (`tools/call`).
"""
from typing import Any, Dict, List, Optional
import json
import sys
import os

# Import MCPServer from sibling lab
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01-mcp-server")))
try:
    from server import MCPServer
except ImportError:
    pass


class MCPClient:
    """
    Connects to MCP servers and coordinates tool execution for LLMs.
    """
    def __init__(self):
        self.connected_servers: Dict[str, Any] = {}
        self.tool_to_server_map: Dict[str, str] = {}
        self.discovered_tools: List[Dict[str, Any]] = []
        self._request_counter = 0

    def connect_server(self, server_id: str, server_instance: Any) -> Dict[str, Any]:
        """
        Performs the MCP handshake and indexes server tools.
        """
        self.connected_servers[server_id] = server_instance
        self._request_counter += 1
        
        # 1. Initialize
        init_res = server_instance.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": self._request_counter,
            "method": "initialize",
            "params": {"clientInfo": {"name": "antigravity-mcp-client", "version": "1.0.0"}}
        })
        
        # 2. Discover Tools
        self._request_counter += 1
        tools_res = server_instance.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": self._request_counter,
            "method": "tools/list",
            "params": {}
        })
        
        tools = tools_res.get("result", {}).get("tools", [])
        for tool in tools:
            name = tool["name"]
            self.tool_to_server_map[name] = server_id
            self.discovered_tools.append(tool)

        return init_res.get("result", {})

    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Converts discovered MCP tools into OpenAI Function Calling format."""
        openai_tools = []
        for t in self.discovered_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"]
                }
            })
        return openai_tools

    def get_anthropic_tool_definitions(self) -> List[Dict[str, Any]]:
        """Converts discovered MCP tools into Anthropic Tool Use format."""
        anthropic_tools = []
        for t in self.discovered_tools:
            anthropic_tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["inputSchema"]
            })
        return anthropic_tools

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Dispatches tool call to the owning MCP Server and returns output string.
        """
        if tool_name not in self.tool_to_server_map:
            raise ValueError(f"No connected MCP server provides tool: {tool_name}")

        server_id = self.tool_to_server_map[tool_name]
        server = self.connected_servers[server_id]
        
        self._request_counter += 1
        res = server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": self._request_counter,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        })
        
        result_payload = res.get("result", {})
        content_items = result_payload.get("content", [])
        return "\n".join(item.get("text", "") for item in content_items)


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🤖 MCP CLIENT & TOOL DISPATCH ENGINE ===\n")

    # Set up mock MCP Servers
    s1 = MCPServer(name="math-mcp", version="1.0")
    s1.register_tool(
        name="power",
        description="Calculates base raised to power exponent.",
        input_schema={"type": "object", "properties": {"base": {"type": "number"}, "exp": {"type": "number"}}},
        handler=lambda base, exp: base ** exp
    )

    s2 = MCPServer(name="system-mcp", version="1.0")
    s2.register_tool(
        name="get_hostname",
        description="Returns system hostname.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: "production-node-01"
    )

    client = MCPClient()
    print("1. Connecting to 'math-mcp' and 'system-mcp' servers...")
    client.connect_server("math", s1)
    client.connect_server("sys", s2)

    print(f"   Connected servers: {list(client.connected_servers.keys())}")
    print(f"   Discovered tools: {[t['name'] for t in client.discovered_tools]}")

    print("\n2. Translated Anthropic Tool Definitions for Claude API:")
    print(json.dumps(client.get_anthropic_tool_definitions(), indent=2))

    print("\n3. Simulating Agent Tool Execution (`power(base=2, exp=10)`):")
    result = client.execute_tool_call("power", {"base": 2, "exp": 10})
    print(f"   Tool Execution Response from math-mcp: {result}")

    print("\n4. Simulating Agent Tool Execution (`get_hostname()`):")
    res2 = client.execute_tool_call("get_hostname", {})
    print(f"   Tool Execution Response from system-mcp: {res2}")
