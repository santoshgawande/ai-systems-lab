# Lab 02 — Agent-as-Tool

Wrap specialised agents as callable tools. The main agent decides when to delegate.

## What you learn

- Defining a subagent as a tool schema (OpenAI function format)
- Main agent loop with tool dispatch → subagent → result → continue
- How the LLM decides WHICH agent to call based on the task
- Difference from orchestrator pattern: dynamic routing vs pre-planned stages

## Run

```bash
pip install httpx
export OPENAI_API_KEY=sk-...    # recommended for tool_calls
python agent_as_tool.py
```

## Pattern

```python
TOOLS = [
    {"type": "function", "function": {
        "name": "research_agent",
        "description": "Research a topic and return key facts.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
    }},
    # ... more agents
]

AGENT_MAP = {
    "research_agent": lambda args: research_agent_fn(args["query"]),
    "code_agent": lambda args: code_agent_fn(args["task"]),
}

while True:
    response = llm(messages, tools=TOOLS)
    if response.finish_reason == "stop":
        return response.content
    for call in response.tool_calls:
        result = AGENT_MAP[call.function.name](parse(call.arguments))
        messages.append({"role": "tool", "content": result})
```

## Orchestrator vs Agent-as-Tool

| | Orchestrator-Subagent | Agent-as-Tool |
|--|--|--|
| Planning | Explicit, upfront | Dynamic, LLM decides |
| Routing | Hard-coded stages | LLM chooses when/which |
| Flexibility | Fixed pipeline | Adapts to any task |
| Predictability | High | Medium |
| Best for | Known workflows | Open-ended tasks |
