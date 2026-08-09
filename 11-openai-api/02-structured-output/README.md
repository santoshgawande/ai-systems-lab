# Lab 02 — OpenAI Structured Outputs

Guarantee valid JSON from the model by providing a strict JSON schema.

## What you learn

- `response_format: {type: "json_schema"}` — model always returns schema-valid JSON
- How to define strict schemas with `additionalProperties: false` and `required` on all fields
- `json_object` mode vs `json_schema` mode — what each guarantees
- How to extract structured data from unstructured text
- Refusal handling — when the model refuses, what the API returns

## Run

```bash
export OPENAI_API_KEY=sk-...
python structured.py
```

## json_schema mode (recommended)

```python
schema = {
    "name": "product_review",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "score": {"type": "number"},
            "summary": {"type": "string"},
            "pros": {"type": "array", "items": {"type": "string"}},
            "cons": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["sentiment", "score", "summary", "pros", "cons"],
        "additionalProperties": false
    }
}

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Analyze: {review_text}"}],
    response_format={"type": "json_schema", "json_schema": schema}
)

result = json.loads(response.choices[0].message.content)
```

## json_object mode (simpler, less strict)

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Respond only with valid JSON."},
        {"role": "user", "content": "Extract name and age from: John is 30 years old."}
    ],
    response_format={"type": "json_object"}
)
# Guarantees parseable JSON, but no schema enforcement
```

## Comparison

| Mode | Parseable JSON | Schema validated | Use when |
|---|---|---|---|
| `json_object` | Yes | No | You just want valid JSON |
| `json_schema` | Yes | Yes (strict) | You need exact field names/types |
| No format | No | No | Free text answers |
