# Lab 01 — Instructor Basics

Extract typed Python objects from LLM responses. No JSON parsing. No regex. Just Pydantic models.

## What you learn

- `instructor.from_anthropic()` and `instructor.from_openai()` — wrap any client
- Define what you want with Pydantic: you always get back a validated Python object
- Nested models, lists, optional fields, enums
- How instructor works under the hood (tool use + JSON schema)

## Run

```bash
pip install instructor anthropic openai pydantic
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY
python instructor_basics.py
```

## Core pattern

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

client = instructor.from_anthropic(Anthropic())

class ExtractedTask(BaseModel):
    title: str
    priority: Literal["high", "medium", "low"]
    due_date: str | None
    assignee: str | None

task = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{
        "role": "user",
        "content": "Fix the critical login bug ASAP, assign to Alice"
    }],
    response_model=ExtractedTask,
)

print(task.title)     # "Fix the critical login bug"
print(task.priority)  # "high"
print(task.assignee)  # "Alice"
```

## How it works under the hood

1. instructor converts your Pydantic model to a JSON schema
2. Sends it as a tool definition to the LLM
3. LLM calls the tool with arguments matching your schema
4. instructor parses and validates the arguments into your Pydantic class
5. Returns the fully-typed object

This is why it's reliable: the LLM is filling in a form, not writing freeform JSON.
