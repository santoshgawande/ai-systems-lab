# Lab 02 — OpenAI Moderation API

Fast, free pre-filter for harmful content before your main LLM call.

## What you learn

- `client.moderations.create()` — all harm categories with confidence scores
- Integration pattern: moderate input → call LLM only if clean
- Threshold tuning: when to flag vs when to let through
- `omni-moderation-latest` vs `text-moderation-latest`

## Run

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python moderation.py
```

## Key API

```python
from openai import OpenAI
client = OpenAI()

result = client.moderations.create(
    model="omni-moderation-latest",
    input="the text to check",
)
r = result.results[0]

print(r.flagged)                        # True/False overall
print(r.categories.violence)            # True if violence detected
print(r.category_scores.violence)       # 0.0 - 1.0 confidence score
```

## Safe LLM pipeline

```python
def safe_complete(user_input: str) -> str:
    mod = client.moderations.create(input=user_input)
    if mod.results[0].flagged:
        return "I can't help with that."
    return llm_complete(user_input)    # only reach here if clean
```

## Categories

| Category | Detects |
|----------|---------|
| `harassment` | Bullying, threats |
| `hate` | Protected-characteristic hatred |
| `self-harm` | Suicide/self-injury content |
| `sexual` | Explicit sexual content |
| `violence` | Violent content |
| + `/threatening`, `/intent`, `/graphic` sub-categories | More specific |

## Cost and speed

- **Free** — no charge per moderation call
- **Fast** — ~100-200ms, run synchronously before your LLM call
- Use `omni-moderation-latest` (newer, better recall than `text-moderation-latest`)
