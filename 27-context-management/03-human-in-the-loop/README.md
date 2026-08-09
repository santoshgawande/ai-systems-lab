# Lab 03 — Human-in-the-Loop

Pause agent execution for human input, approval, or clarification.

## What you learn

- Uncertainty detection: LLM self-assesses confidence, asks clarifying questions
- Approval gates: pause before risky/irreversible actions
- Interrupt-and-resume: serialise agent state, resume after human response
- Production integrations: Slack, email, web UI, LangGraph, Temporal

## Run

```bash
pip install httpx
python human_in_the_loop.py
```

## Three patterns

### 1. Uncertainty detection

```python
def maybe_ask(task: str) -> str | None:
    result = llm(f"Confidence 0-1 for: {task}. JSON: {{confidence, questions}}")
    if result["confidence"] < 0.7:
        return result["questions"][0]  # ask the user
    return None                         # proceed
```

### 2. Approval gate

```python
def approval_gate(action, details) -> bool:
    print(f"Proposed: {action}\nDetails: {details}")
    return input("Approve? [y/N]: ").lower() == "y"

if action.is_risky:
    if not approval_gate(action, details):
        agent.stop()
```

### 3. Interrupt-and-resume

```python
# Agent serialises state at each step
checkpoint = {"goal": ..., "steps_done": [...], "status": "paused"}
db.save(checkpoint)

# Human reviews, approves via Slack/email
# Agent resumes:
agent = Agent.from_checkpoint(db.load(checkpoint_id))
agent.continue()
```

## When to require human approval

| Action | Require approval? |
|--------|------------------|
| Read/query data | No |
| Write/create (reversible) | Maybe |
| Delete/drop (irreversible) | Yes |
| Send message/email | Yes |
| Deploy to production | Yes |
| Spend money (API/cloud) | Yes (above threshold) |

## Production integrations

- **LangGraph**: built-in `interrupt()` / `Command(resume=...)` primitives
- **Temporal.io**: durable workflows with human signal channels
- **Slack bot**: post proposed action → wait for emoji reaction
