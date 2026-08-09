# Lab 02 — Summarisation-Based Context

Compress old conversation turns into a running summary instead of discarding them.

## What you learn

- Trigger-based compression: summarise when history exceeds N tokens
- Injecting the summary as a `system` message for future turns
- Preserving key facts (user preferences, decisions) across compression
- Sliding window vs summarisation trade-offs

## Run

```bash
pip install httpx tiktoken
python summarization.py
# Adjust trigger: SUMMARY_TRIGGER_TOKENS=2000 python summarization.py
```

## Pattern

```python
def chat(self, user_input):
    self.history.append({"role": "user", "content": user_input})

    if token_count(self.history) > self.trigger:
        old = self.history[:-4]         # all but last 2 turns
        piece = summarise_llm(old)      # LLM → bullet points
        self.summary += piece
        self.history = self.history[-4:]  # keep only recent

    messages = [
        system_prompt,
        {"role": "system", "content": f"Earlier: {self.summary}"},
        *self.history,                  # verbatim recent turns
    ]
    return llm(messages)
```

## Sliding window vs summarisation

| | Sliding Window | Summarisation |
|--|--|--|
| Old info | Lost completely | Preserved as summary |
| Extra LLM calls | None | 1 per compression |
| Complexity | Low | Medium |
| Best for | Independent Q&A | Cumulative conversations |
| Latency impact | None | +100-500ms when triggered |

## When summarisation wins

- User stated their tech stack in turn 1 — should inform turn 20 answers
- Multi-step debugging: earlier error context matters later
- Interview/coaching: preferences and goals set early persist throughout
