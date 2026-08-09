# mini-claude-code

A minimal CLI agent that reads/writes files and runs shell commands — a simplified version of what Claude Code does internally.

## What you build

- ReAct loop (Thought → Action → Observation → repeat)
- 4 tools: `read_file`, `write_file`, `list_dir`, `run_bash`
- Streaming output token-by-token
- Safety rules: blocks destructive commands

## Run

```bash
python agent.py "list all python files in the current directory"
python agent.py "write a hello world script to /tmp/hello.py and run it"
python agent.py "read /etc/hosts and count how many non-comment lines there are"
python agent.py "create a file /tmp/fizzbuzz.py that prints fizzbuzz 1-20, then run it"
```

## Architecture

```
User task
  → System prompt with tool definitions
  → LLM stream response (Thought/Action/Action Input)
  → Parse structured output
  → Execute tool locally
  → Append "Observation: <result>" as next user message
  → Repeat until "Final Answer:"
```

## What this teaches

This is the core of every agentic system — Claude Code, Devin, OpenAI Assistants.
The "intelligence" is the LLM. The loop, parsing, and tool execution is just ~100 lines of Python.
