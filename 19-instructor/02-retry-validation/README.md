# Lab 02 — Retry & Validation

Add Pydantic validators to enforce constraints the LLM might violate — instructor auto-retries with the error as feedback.

## What you learn

- `@field_validator` — enforce business rules on extracted fields
- Automatic retry: instructor sends the validation error back to the LLM
- `max_retries` — how many times instructor will attempt correction
- When to validate vs when to trust the model

## Run

```bash
python retry_validation.py
```

## Retry loop

```
1. LLM returns JSON
2. Pydantic validates against model
3. Validator fails (e.g. score out of range)
4. instructor adds error to messages:
   "Validation Error: score must be between 1 and 10. Got: 15"
5. LLM sees its mistake and corrects
6. Retry from step 1 (up to max_retries times)
```

## Example validators

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float
    tags: list[str]

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v):
        if not 1 <= v <= 10:
            raise ValueError(f"score must be 1-10, got {v}")
        return round(v, 1)

    @field_validator("tags")
    @classmethod
    def max_five_tags(cls, v):
        if len(v) > 5:
            raise ValueError(f"max 5 tags, got {len(v)}")
        return v[:5]

    @field_validator("sentiment")
    @classmethod
    def sentiment_matches_score(cls, v, info):
        score = info.data.get("score", 5)
        if v == "positive" and score < 6:
            raise ValueError(f"positive sentiment requires score >= 6, got {score}")
        return v
```
