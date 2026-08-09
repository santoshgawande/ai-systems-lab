# Lab 01 — Tool Use / Function Calling

Define tools as JSON schema, have the model decide which to call, execute them, and feed results back.

## What you learn

- How tool definitions inflate the prompt (each tool adds ~100 tokens)
- How the model signals a tool call in its response
- The request → tool call → result → final answer loop
- Why parallel tool calls matter for latency

## Run

```bash
python tools.py
```

## The tool use protocol

```
1. You define tools as JSON schema in the request
2. Model responds with a tool_call instead of text
3. You execute the tool locally
4. You append the result as a "tool" role message
5. Model reads the result and continues → final text answer
```

Tools registered = tokens consumed on every request, even if no tool is called.
Keep tool definitions short and only register tools the model actually needs.
