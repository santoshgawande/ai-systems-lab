# Section 26 — Multi-Agent Patterns

Coordinate multiple LLM agents to solve problems that are too complex for a single prompt.

## What you learn

- Orchestrator-subagent: one planner dispatches specialist workers
- Agent-as-tool: an agent wrapped as a callable tool for another agent
- Shared state, handoff contracts, and result aggregation
- When multi-agent wins over a single large prompt

## Labs

| Lab | What it covers |
|---|---|
| 01-orchestrator-subagent | Planner → specialist dispatch, result aggregation |
| 02-agent-as-tool | Wrap a research agent as a callable tool |

## Setup

```bash
pip install -r requirements.txt
```

## Pattern overview

```
Orchestrator-Subagent:          Agent-as-Tool:
  User                            User
    ↓                               ↓
  Orchestrator ←plan→           Main Agent
    ↓      ↓                      ↓  (calls tools)
  Sub A  Sub B                  research_agent()
    ↓      ↓                    code_agent()
  Orchestrator ←results→        Main Agent
    ↓                               ↓
  User                            User
```

## When multi-agent patterns help

- Parallelising independent sub-tasks (3x faster than sequential)
- Specialists beat generalists: a coding agent + a research agent outperforms one that does both
- Context window management: each subagent gets a focused context
- Separation of concerns: each agent has its own system prompt and tools
