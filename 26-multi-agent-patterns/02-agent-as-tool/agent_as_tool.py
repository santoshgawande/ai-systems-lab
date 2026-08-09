"""
Agent-as-Tool pattern: wrap a specialised agent as a callable tool.
The main agent decides WHEN to call sub-agents via tool_calls, just like any other tool.
"""
import os
import json
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# ─── Subagent implementations ─────────────────────────────────────────────────

def research_agent_fn(query: str) -> str:
    """A research subagent: finds facts about a topic."""
    prompt = f"Provide 3 concrete facts about: {query}. Be specific and concise."
    return _llm(prompt, "You are a factual research assistant. Cite specifics.")


def code_agent_fn(task: str, language: str = "python") -> str:
    """A coding subagent: writes code for a given task."""
    prompt = f"Write a {language} function for: {task}\nInclude a brief docstring."
    return _llm(prompt, "You are an expert programmer. Write clean, idiomatic code.")


def calculator_agent_fn(expression: str) -> str:
    """A calculation subagent: evaluates math expressions safely."""
    try:
        # Safe eval: only allow numbers and basic operators
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only arithmetic expressions allowed"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _llm(prompt: str, system: str = "") -> str:
    if OPENAI_KEY:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": 300},
            timeout=30,
        )
        return r.json()["choices"][0]["message"]["content"].strip()

    if ANTHROPIC_KEY:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "system": system or "You are a helpful assistant.",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        return r.json()["content"][0]["text"].strip()

    full = f"{system}\n\n{prompt}".strip() if system else prompt
    r = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": full, "stream": False},
        timeout=60,
    )
    return r.json()["response"].strip()


# ─── Tool definitions (OpenAI tool_call format) ───────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research_agent",
            "description": "Research a topic and return key facts. Use when you need factual background information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic to research"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_agent",
            "description": "Write code for a programming task. Use when the user needs working code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What the code should do"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator_agent",
            "description": "Evaluate arithmetic expressions. Use for any calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression to evaluate"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_MAP = {
    "research_agent": lambda args: research_agent_fn(args["query"]),
    "code_agent": lambda args: code_agent_fn(args["task"], args.get("language", "python")),
    "calculator_agent": lambda args: calculator_agent_fn(args["expression"]),
}


# ─── Main agent loop (OpenAI tool_calls) ─────────────────────────────────────

def run_main_agent(user_message: str) -> str:
    if not OPENAI_KEY:
        return _run_without_tool_calls(user_message)

    messages = [{"role": "user", "content": user_message}]
    print(f"  [Main agent] Processing: {user_message[:60]!r}")

    while True:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 400,
            },
            timeout=30,
        )
        resp = r.json()
        choice = resp["choices"][0]
        msg = choice["message"]

        if choice["finish_reason"] == "stop":
            return msg["content"]

        if choice["finish_reason"] == "tool_calls":
            messages.append(msg)
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                print(f"  [Main agent] → calling {fn_name}({fn_args})")

                result = TOOL_MAP[fn_name](fn_args)
                print(f"  [{fn_name}] returned {len(result)} chars")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })


def _run_without_tool_calls(user_message: str) -> str:
    """Fallback: manually route to agents based on keywords."""
    print("  [No OpenAI key — using keyword routing fallback]")
    lower = user_message.lower()
    parts = []

    if any(w in lower for w in ["research", "what is", "tell me about", "facts"]):
        topic = user_message
        result = research_agent_fn(topic)
        parts.append(f"Research: {result}")

    if any(w in lower for w in ["code", "function", "write", "implement"]):
        result = code_agent_fn(user_message)
        parts.append(f"Code: {result}")

    if any(c in user_message for c in "0123456789") and any(c in user_message for c in "+-*/"):
        import re
        expr_match = re.search(r"[\d\s\+\-\*/\(\)\.]+", user_message)
        if expr_match:
            result = calculator_agent_fn(expr_match.group().strip())
            parts.append(f"Calculation: {result}")

    if not parts:
        return _llm(user_message)

    return "\n\n".join(parts)


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== AGENT-AS-TOOL DEMO ===\n")

try:
    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    ollama_ok = True
except Exception:
    ollama_ok = False

if not OPENAI_KEY and not ANTHROPIC_KEY and not ollama_ok:
    print("No LLM available. Start Ollama or set OPENAI_API_KEY\n")
    print("""
Agent-as-Tool pattern:

# Define subagents as tool schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research_agent",
            "description": "Research a topic...",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    },
    # ... more agents as tools
]

# Main agent loop
while True:
    response = llm(messages, tools=TOOLS)

    if response.finish_reason == "stop":
        return response.content        # done

    if response.finish_reason == "tool_calls":
        for call in response.tool_calls:
            result = AGENT_MAP[call.function.name](call.function.arguments)
            messages.append({"role": "tool", "content": result})
        # continue loop — agent sees tool results and decides next action

Key insight:
  - The main agent DECIDES when to delegate — no hard-coded routing
  - Subagents are just functions from the main agent's perspective
  - Works with OpenAI tool_calls, Anthropic tool_use, or any function-calling LLM
""")
    raise SystemExit(0)

test_messages = [
    "What is vector similarity search and give me a Python function to compute cosine similarity.",
    "Research transformer architecture and calculate how many parameters a model with 96 layers, hidden_size=12288, and 96 heads would have approximately: 96 * 12288 * 12288 * 4.",
]

for msg in test_messages:
    print(f"{'─'*60}")
    print(f"User: {msg}\n")
    result = run_main_agent(msg)
    print(f"\nFinal answer:\n{result}")
    print()
