# Lab 03 — Multi-Agent Systems

One LLM plans, another executes. The pattern behind Claude's subagents and AutoGPT.

## What you learn

- How to separate planning from execution using two LLM calls
- How to pass context between agents via structured messages
- Why specialization improves output quality (planner focuses on decomposition, executor on completion)
- The "agent-as-tool" pattern: the orchestrator treats the executor like a function call

## Run

```bash
python orchestrator.py "explain the trade-offs between RAG and fine-tuning"
python orchestrator.py "write a Python function to parse JSON with error handling"
```

## Architecture

```
User → Orchestrator (Planner)
          ↓  breaks into steps
       [step 1, step 2, step 3]
          ↓  for each step
       Executor (Worker)
          ↓  returns result
       Synthesizer
          ↓  combines all results
       Final Answer → User
```
