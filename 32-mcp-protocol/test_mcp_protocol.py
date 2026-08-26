import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-mcp-server"))
sys.path.insert(0, os.path.join(base_dir, "02-mcp-client"))

from server import MCPServer
from client import MCPClient


class TestMCPProtocol(unittest.TestCase):
    def test_mcp_server_handshake_and_tools(self):
        server = MCPServer(name="test-server", version="1.0")
        server.register_tool(
            name="multiply",
            description="Multiplies two numbers",
            input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
            handler=lambda a, b: a * b
        )
        
        # Test Initialize
        init_res = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_res["result"]["serverInfo"]["name"], "test-server")
        
        # Test Tool Call
        call_res = server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "multiply", "arguments": {"a": 6, "b": 7}}
        })
        self.assertEqual(call_res["result"]["content"][0]["text"], "42")

    def test_mcp_client_multi_server_orchestration(self):
        s1 = MCPServer(name="srv1")
        s1.register_tool("echo", "Echoes input", {"type": "object"}, lambda msg: f"Echo: {msg}")
        
        client = MCPClient()
        client.connect_server("s1", s1)
        
        # Check tool translations
        tools = client.get_openai_tool_definitions()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "echo")
        
        # Execute tool
        out = client.execute_tool_call("echo", {"msg": "hello MCP"})
        self.assertEqual(out, "Echo: hello MCP")


if __name__ == "__main__":
    unittest.main()
