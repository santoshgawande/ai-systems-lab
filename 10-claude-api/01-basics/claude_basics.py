"""
Anthropic Claude API basics.
Covers: Messages API format, system prompts, tool use, streaming.
Requires: ANTHROPIC_API_KEY env var
"""
import os
import json

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("ANTHROPIC_API_KEY not set.")
    print("Get a key at https://console.anthropic.com/")
    print("Then: export ANTHROPIC_API_KEY=sk-ant-...\n")
    print("Showing API shapes without running live calls.\n")
    LIVE = False
else:
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    LIVE = True

MODEL = "claude-sonnet-4-6"

# ─── 1. Basic message ─────────────────────────────────────────────────────────
print("=== 1. BASIC MESSAGE ===")
print("""
# Key difference from OpenAI: 'system' is a separate parameter
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a concise technical assistant.",   # NOT in messages[]
    messages=[
        {"role": "user", "content": "What is a database index?"}
    ]
)
# Access:
print(response.content[0].text)
print(f"Input tokens:  {response.usage.input_tokens}")   # NOT prompt_tokens
print(f"Output tokens: {response.usage.output_tokens}")  # NOT completion_tokens
print(f"Stop reason:   {response.stop_reason}")          # end_turn | tool_use | max_tokens
""")

if LIVE:
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system="You are a concise technical assistant. Answer in 2-3 sentences.",
        messages=[{"role": "user", "content": "What is a database index?"}]
    )
    print(f"Response: {response.content[0].text}")
    print(f"Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out\n")

# ─── 2. Multi-turn conversation ───────────────────────────────────────────────
print("=== 2. MULTI-TURN CONVERSATION ===")
print("""
# Claude requires alternating user/assistant turns (no consecutive same-role)
messages = [
    {"role": "user",      "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language."},
    {"role": "user",      "content": "What is it good for?"},
]
""")

if LIVE:
    messages = [
        {"role": "user",      "content": "What is Python? One sentence."},
        {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for readability."},
        {"role": "user",      "content": "What's it best used for? One sentence."},
    ]
    r = client.messages.create(model=MODEL, max_tokens=128, messages=messages)
    print(f"Turn 3 response: {r.content[0].text}\n")

# ─── 3. Tool use ──────────────────────────────────────────────────────────────
print("=== 3. TOOL USE ===")
print("""
# Define tools with JSON schema (same concept as OpenAI, slightly different format)
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {           # ← 'input_schema' not 'parameters'
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# When Claude wants to call a tool, stop_reason = "tool_use"
if response.stop_reason == "tool_use":
    tool_call = next(b for b in response.content if b.type == "tool_use")
    print(f"Tool: {tool_call.name}, Input: {tool_call.input}")
    # Execute tool, then send result back:
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": [{
        "type": "tool_result",
        "tool_use_id": tool_call.id,
        "content": "Sunny, 22°C"
    }]})
""")

if LIVE:
    tools = [{
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    }]
    messages = [{"role": "user", "content": "What is 25 * 48?"}]
    r = client.messages.create(model=MODEL, max_tokens=256, tools=tools, messages=messages)

    if r.stop_reason == "tool_use":
        tc = next((b for b in r.content if b.type == "tool_use"), None)
        if tc:
            import math
            result = str(eval(tc.input.get("expression", "0"), {"__builtins__": {}}, vars(math)))
            print(f"Tool called: {tc.name}({tc.input}) → {result}")
            messages.append({"role": "assistant", "content": r.content})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tc.id, "content": result}]})
            final = client.messages.create(model=MODEL, max_tokens=128, tools=tools, messages=messages)
            print(f"Final: {final.content[0].text}\n")

# ─── 4. Streaming ─────────────────────────────────────────────────────────────
print("=== 4. STREAMING ===")
print("""
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Explain RAG in 3 sentences."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
""")

if LIVE:
    print("Streaming: ", end="")
    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": "What is RAG? 2 sentences."}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")
