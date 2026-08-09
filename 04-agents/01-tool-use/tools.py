import json
import math
import datetime
import httpx

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use for any arithmetic or math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '2 + 2', 'sqrt(16)', '15 * 847'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
]

MOCK_SEARCH = {
    "python": "Python is a high-level programming language known for simplicity and readability.",
    "ollama": "Ollama is a tool for running large language models locally on your machine.",
    "rag": "RAG (Retrieval-Augmented Generation) grounds LLM responses in retrieved documents.",
    "transformer": "Transformers use self-attention to process sequences in parallel.",
}


def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            safe_globals = {k: v for k, v in vars(math).items() if not k.startswith("_")}
            return str(eval(args["expression"], {"__builtins__": {}}, safe_globals))
        except Exception as e:
            return f"Error: {e}"

    elif name == "get_datetime":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elif name == "search":
        q = args.get("query", "").lower()
        for key, val in MOCK_SEARCH.items():
            if key in q:
                return val
        return f"No results found for: {args.get('query')}"

    return f"Unknown tool: {name}"


def run_agent(user_query: str):
    print(f"User: {user_query}\n")
    messages = [{"role": "user", "content": user_query}]

    for step in range(6):
        r = httpx.post(f"{OLLAMA}/api/chat", json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }, timeout=60)
        r.raise_for_status()

        msg = r.json()["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            print(f"Assistant: {msg['content']}\n")
            return

        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            result = run_tool(name, args)
            print(f"  [Tool] {name}({json.dumps(args)}) → {result}")
            messages.append({"role": "tool", "content": result})


QUERIES = [
    "What is 15% of 847 plus the square root of 144?",
    "What time is it right now, and what day of the week is that?",
    "Search for information about Ollama, then tell me the current date.",
]

for q in QUERIES:
    print("=" * 60)
    run_agent(q)
