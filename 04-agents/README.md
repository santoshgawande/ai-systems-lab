# 04 — Agents

Tool use, ReAct loops, and multi-agent systems. The internals of Claude Code, GitHub Copilot, and ChatGPT plugins.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Ollama at `http://localhost:11434`.

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-tool-use/` | Define tools as JSON schema, parse model's tool calls, feed results back | `python tools.py` |
| `02-react-loop/` | ReAct: Thought → Action → Observation → repeat until done | `python react.py "your task"` |
| `03-multi-agent/` | Planner + executor pattern, message passing between agents | `python orchestrator.py "your task"` |
| `04-memory/` | In-context (sliding window) vs external (file + semantic) memory | `python memory.py` |

## The ReAct loop (how Claude Code works internally)

```
User task
  → Thought:  what do I need to do?
  → Action:   call a tool (read_file, run_bash, search...)
  → Observation: result of the tool
  → Thought:  what did I learn? what next?
  → Action:   ...
  → Final Answer: done
```

## Key concepts

- Tool use is just JSON: model outputs structured JSON, you execute it, feed result back as next message
- An "agent" is an LLM in a loop — the intelligence is the model, the loop is ~20 lines of Python
- Stop conditions are critical: max iterations, "Final Answer" signal, or empty tool call
- Multi-agent = one LLM calling another LLM as if it were a tool (same pattern, nested)
