from __future__ import annotations
"""
Model Context Protocol (MCP) JSON-RPC 2.0 Server.

MCP is the open standard created by Anthropic for safely exposing tools, resources,
and prompts to AI models and coding assistants (e.g. Claude Code, Cursor).

Protocol primitives:
- JSON-RPC 2.0 framing (`jsonrpc: "2.0"`, `id`, `method`, `params`).
- Handshake & Capability negotiation (`initialize`).
- Tool discovery & invocation (`tools/list`, `tools/call`).
- Resource discovery & reading (`resources/list`, `resources/read`).
"""
import json
from typing import Any, Callable, Dict, List, Optional
import dataclasses


@dataclasses.dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]


@dataclasses.dataclass
class MCPResource:
    uri: str
    name: str
    mime_type: str
    content_provider: Callable[[], str]


class MCPServer:
    """
    Standard MCP Server handling JSON-RPC 2.0 requests.
    """
    def __init__(self, name: str = "ai-systems-mcp-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable[..., Any]):
        self.tools[name] = MCPTool(name=name, description=description, input_schema=input_schema, handler=handler)

    def register_resource(self, uri: str, name: str, mime_type: str, content_provider: Callable[[], str]):
        self.resources[uri] = MCPResource(uri=uri, name=name, mime_type=mime_type, content_provider=content_provider)

    def handle_raw_request(self, raw_json: str) -> str:
        try:
            req = json.loads(raw_json)
        except Exception:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            })

        response = self.handle_jsonrpc(req)
        return json.dumps(response, indent=2)

    def handle_jsonrpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if req.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: expected jsonrpc 2.0"}
            }

        # 1. Initialize Handshake
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.name, "version": self.version},
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"subscribe": False, "listChanged": True}
                    }
                }
            }

        # 2. Tools List
        elif method == "tools/list":
            tools_list = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema
                }
                for t in self.tools.values()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}}

        # 3. Tool Execution Call
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            if tool_name not in self.tools:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
                }
            tool = self.tools[tool_name]
            try:
                result_content = tool.handler(**arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(result_content)}],
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Execution error: {str(e)}"}],
                        "isError": True
                    }
                }

        # 4. Resources List & Read
        elif method == "resources/list":
            res_list = [
                {"uri": r.uri, "name": r.name, "mimeType": r.mime_type}
                for r in self.resources.values()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": res_list}}

        elif method == "resources/read":
            uri = params.get("uri")
            if uri not in self.resources:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Resource not found: {uri}"}
                }
            res = self.resources[uri]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [{
                        "uri": res.uri,
                        "mimeType": res.mime_type,
                        "text": res.content_provider()
                    }]
                }
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🔌 MODEL CONTEXT PROTOCOL (MCP) SERVER DEMO ===\n")

    server = MCPServer(name="finance-analytics-mcp", version="1.2.0")

    # Register a tool
    server.register_tool(
        name="calculate_compound_interest",
        description="Calculates compound interest given principal, annual rate, and years.",
        input_schema={
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "rate": {"type": "number"},
                "years": {"type": "integer"}
            },
            "required": ["principal", "rate", "years"]
        },
        handler=lambda principal, rate, years: principal * ((1 + rate) ** years)
    )

    # Register a resource
    server.register_resource(
        uri="repo://config/system.json",
        name="System Configuration",
        mime_type="application/json",
        content_provider=lambda: json.dumps({"environment": "production", "region": "ap-south-1"})
    )

    # Simulate client interactions
    print("1. Handshake Initialize Request:")
    init_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    print(server.handle_raw_request(init_req))

    print("\n2. Discover Tools Request (`tools/list`):")
    tools_req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    print(server.handle_raw_request(tools_req))

    print("\n3. Invoke Tool Request (`tools/call`):")
    call_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_compound_interest",
            "arguments": {"principal": 100000, "rate": 0.12, "years": 5}
        }
    })
    print(server.handle_raw_request(call_req))
