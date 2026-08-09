# Lab 01 — OpenAI Function Calling

How OpenAI models request tool calls and how you dispatch them.

## What you learn

- How to define tools as JSON schemas in the `tools` array
- How the model signals a tool call via `finish_reason: "tool_calls"`
- How to dispatch tool calls and feed `tool` role messages back
- Parallel function calling — model invokes multiple tools in one turn
- How to implement a complete tool loop (call → dispatch → observe → continue)

## Run

```bash
export OPENAI_API_KEY=sk-...
python functions.py
```

## Tool call flow

```
User message
  → model decides which tool(s) to call
  → assistant message with tool_calls[] (no text yet)
  → you execute each tool
  → tool messages with results (role: "tool")
  → model reads results, produces final answer
```

## API shape

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo and Paris?"}],
    tools=tools,
    tool_choice="auto"   # "auto" | "required" | "none" | specific tool
)

# If model wants tools:
if response.choices[0].finish_reason == "tool_calls":
    for tool_call in response.choices[0].message.tool_calls:
        print(tool_call.function.name)       # "get_weather"
        print(tool_call.function.arguments)  # '{"city": "Tokyo", "unit": "celsius"}'
```

## Key differences from Claude

| Feature | OpenAI | Anthropic |
|---|---|---|
| Tool result role | `"tool"` | `"user"` (nested) |
| Tool call ID | `tool_call_id` | `tool_use_id` |
| Parallel calls | multiple `tool_calls[]` | multiple `tool_use` blocks |
| Force tool use | `tool_choice: "required"` | `tool_choice: {"type": "any"}` |
