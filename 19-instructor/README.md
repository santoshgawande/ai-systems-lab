# Section 19 — Instructor (Pydantic + LLMs)

Extract typed Python objects from LLM responses with automatic validation and retry.

## What you learn

- `instructor` library — wraps any LLM client to return Pydantic models
- Pydantic validators on LLM output — enforce constraints the model might violate
- Automatic retry on validation failure — model self-corrects with the error as feedback
- Nested models, unions, lists — handle complex extraction schemas

## Labs

| Lab | What it covers |
|---|---|
| 01-basics | instructor with Claude + OpenAI, return typed Pydantic objects |
| 02-retry-validation | Pydantic field_validator, automatic retry loop, correction feedback |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY
```

## Why instructor

Without instructor:
```python
# Fragile — model may return wrong fields, wrong types, extra text
response = client.messages.create(...)
data = json.loads(response.content[0].text)  # may throw
```

With instructor:
```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

client = instructor.from_anthropic(Anthropic())

class User(BaseModel):
    name: str
    age: int
    email: str

# Returns a User object — guaranteed valid
user = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": "Extract: John Doe, 30, john@example.com"}],
    response_model=User,
)
print(user.name, user.age)  # type-safe!
```

## Provider support

- `instructor.from_anthropic(Anthropic())` — Claude
- `instructor.from_openai(OpenAI())` — GPT-4o, GPT-4o-mini
- `instructor.from_gemini(genai.GenerativeModel(...))` — Gemini
- Also: Cohere, Mistral, Ollama (via OpenAI-compatible endpoint)
