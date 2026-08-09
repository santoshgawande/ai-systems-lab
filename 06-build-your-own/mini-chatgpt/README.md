# mini-chatgpt

A streaming multi-turn chat with context window management and persistent session history.

## What you build

- Streaming token output (like ChatGPT)
- Sliding window context (trims old turns to fit token budget)
- Persistent history across restarts (saved to `/tmp/mini_chatgpt_history.json`)
- Real latency + token usage displayed after each response

## Run

```bash
python chat.py
```

### Commands

| Command | Action |
|---|---|
| `quit` | Exit |
| `/clear` | Clear conversation history |
| `/history` | Show recent messages |

## What this teaches

Multi-turn chat is just appending to a `messages` array.
The hard part is keeping it within the context window as it grows.
Sliding window (drop oldest turns) is the simplest strategy — summarization is smarter but more complex.

## Context management

```
Full history: [turn1, turn2, turn3, turn4, turn5, turn6]

If total tokens > MAX_CONTEXT_TOKENS:
  Drop oldest turns until it fits
  → [turn4, turn5, turn6]

Model only sees the recent window — but to the user it feels continuous.
```
