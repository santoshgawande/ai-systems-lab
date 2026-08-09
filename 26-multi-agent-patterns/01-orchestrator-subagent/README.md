# Lab 01 — Orchestrator-Subagent

A planner LLM breaks down a task and dispatches specialist agents in parallel.

## What you learn

- Dependency mapping: what must run before what
- `asyncio.gather()` for parallel subagent dispatch
- Each subagent has its own system prompt and specialisation
- Orchestrator owns the plan; subagents own execution

## Run

```bash
pip install httpx
python orchestrator.py
# Works with Ollama, OpenAI, or Anthropic
```

## Pattern

```python
async def orchestrator(task):
    # Stage 1: blocking (writer needs research output)
    research = await research_agent(task)

    # Stage 2: parallel (writer and critic are independent)
    writer, critic = await asyncio.gather(
        writer_agent(task, research.output),
        critic_agent(research.output),
    )

    # Stage 3: synthesise all results
    return synthesise(research, writer, critic)
```

## When to use

- Task decomposes naturally into 2+ specialist subtasks
- Independent subtasks can run in parallel → 2-3x speedup
- Each stage produces output that feeds the next
- Single-agent context window would be too large

## Agent design rules

1. **Each agent gets one job** — researcher doesn't write; writer doesn't critique
2. **Separate system prompts** — specialise tone and constraints per agent
3. **Typed handoffs** — define what each agent accepts and returns
4. **Orchestrator is not a worker** — it plans and aggregates only
