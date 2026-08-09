# Lab 04 — Memory Systems

In-context (sliding window), external (file store), and semantic (vector) memory for persistent agents.

## What you learn

- **In-context memory**: conversation history trimmed to fit the context window (sliding window)
- **External memory**: facts stored to disk and loaded in future sessions (persistent)
- **Semantic memory**: retrieved by embedding similarity — only relevant facts are injected

## Run

```bash
python memory.py
```

Type `memory` to see stored facts.
Type `quit` to exit.
Restart and it remembers you.

## Memory architecture

```
User message
  ↓
Semantic search → find relevant past facts → inject into system prompt
  ↓
Sliding window  → last 6 messages as context
  ↓
LLM generates response
  ↓
Extract facts from exchange → store to disk + embed for future retrieval
```

## Key insight

Full conversation history grows forever. Sliding window keeps recent context.
Semantic memory surfaces the RIGHT old facts without loading everything.
This is how Claude Code keeps context across tool calls without hitting token limits.
