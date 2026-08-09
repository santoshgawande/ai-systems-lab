"""
OpenAI Assistants API: create assistant → thread → run → poll → read response.
Shows basic Q&A flow and tool-calling within a run (requires_action).
Requires: OPENAI_API_KEY
"""
import os
import json
import time
import math

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("OPENAI_API_KEY not set. Showing Assistants API mechanics.\n")
    LIVE = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    LIVE = True

MODEL = "gpt-4o-mini"

# ─── Tools for the assistant ─────────────────────────────────────────────────

ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. 'sqrt(144)', '2**10'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_formula",
            "description": "Look up a mathematical formula by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Formula name, e.g. 'quadratic', 'pythagorean'"}
                },
                "required": ["name"]
            }
        }
    }
]

FORMULAS = {
    "quadratic": "x = (-b ± sqrt(b²-4ac)) / 2a  for ax²+bx+c=0",
    "pythagorean": "a² + b² = c²",
    "distance": "d = sqrt((x2-x1)² + (y2-y1)²)",
    "compound_interest": "A = P(1 + r/n)^(nt)",
    "area_circle": "A = πr²",
    "slope": "m = (y2-y1) / (x2-x1)",
}


def execute_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            safe = {k: v for k, v in vars(math).items() if not k.startswith("_")}
            return str(eval(args["expression"], {"__builtins__": {}}, safe))
        except Exception as e:
            return f"Error: {e}"
    elif name == "lookup_formula":
        key = args["name"].lower().replace(" ", "_")
        return FORMULAS.get(key, f"Formula '{args['name']}' not found in database.")
    return f"Unknown tool: {name}"


# ─── Run poller with tool execution ──────────────────────────────────────────

def poll_run(thread_id: str, run_id: str, timeout: int = 60) -> str:
    """Poll until run completes, handling tool calls along the way."""
    deadline = time.time() + timeout

    while time.time() < deadline:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

        if run.status == "completed":
            return "completed"

        elif run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            print(f"  [Run requires action: {len(tool_calls)} tool call(s)]")

            outputs = []
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                print(f"  Tool: {tc.function.name}({args}) → {result}")
                outputs.append({"tool_call_id": tc.id, "output": result})

            # Submit all tool outputs at once
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=outputs
            )

        elif run.status in ("failed", "cancelled", "expired"):
            print(f"  [Run ended with status: {run.status}]")
            if run.last_error:
                print(f"  Error: {run.last_error.message}")
            return run.status

        time.sleep(1)

    return "timeout"


def get_latest_response(thread_id: str) -> str:
    """Get the most recent assistant message from the thread."""
    messages = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
    if not messages.data:
        return ""
    msg = messages.data[0]
    if msg.role != "assistant":
        return ""
    parts = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text.value)
    return "\n".join(parts)


def ask(thread_id: str, assistant_id: str, question: str) -> str:
    """Add a user message, run the assistant, return the response."""
    print(f"User: {question}")

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    status = poll_run(thread_id, run.id)
    if status != "completed":
        return f"[Run failed: {status}]"

    answer = get_latest_response(thread_id)
    print(f"Assistant: {answer}\n")
    return answer


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== OPENAI ASSISTANTS API DEMO ===\n")

if not LIVE:
    print("Assistants API lifecycle:\n")
    print("""
# Step 1: Create assistant (do this once, store the ID)
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="Help students understand math. Show your work.",
    model="gpt-4o-mini",
    tools=[{"type": "function", "function": {...}}]
)
# assistant.id = "asst_abc123" — reuse this in production

# Step 2: Create a thread per conversation
thread = client.beta.threads.create()

# Step 3: Add messages
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve: 2x² + 5x - 3 = 0"
)

# Step 4: Run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Step 5: Poll
while run.status in ("queued", "in_progress", "requires_action"):
    if run.status == "requires_action":
        # Execute tools, submit outputs (see functions.py)
        ...
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

# Step 6: Read response
messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
print(messages.data[0].content[0].text.value)
""")
    print("Run status flow:")
    print("  queued → in_progress → completed  (happy path)")
    print("  queued → in_progress → requires_action → in_progress → completed  (tool use)")
    print("  queued → in_progress → failed  (error)")
    print("\nKey differences from Chat Completions:")
    print("  - Thread persists across multiple turns (no message management)")
    print("  - Run is an async operation (must poll)")
    print("  - built-in tools: 'code_interpreter', 'file_search' (no implementation needed)")
else:
    # Create assistant
    print("Creating assistant...")
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions=(
            "You are a helpful math tutor. When solving problems, show your reasoning. "
            "Use the calculator tool for arithmetic and the lookup_formula tool when formulas are needed."
        ),
        model=MODEL,
        tools=ASSISTANT_TOOLS
    )
    print(f"Assistant created: {assistant.id}\n")

    # Create thread
    thread = client.beta.threads.create()
    print(f"Thread created: {thread.id}\n")
    print("-" * 60)

    # Multi-turn conversation
    questions = [
        "What is sqrt(1764) + 2**12?",
        "What's the quadratic formula? Then solve 2x² + 5x - 3 = 0.",
        "Now, using the distance formula, what's the distance between (1,2) and (4,6)?",
    ]

    for q in questions:
        ask(thread_id=thread.id, assistant_id=assistant.id, question=q)

    # Clean up — delete the assistant (optional; stored remotely otherwise)
    client.beta.assistants.delete(assistant.id)
    print(f"(Assistant {assistant.id} deleted)")
