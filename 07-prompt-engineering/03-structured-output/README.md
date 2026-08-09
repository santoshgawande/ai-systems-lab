# Lab 03 — Structured Output

Getting LLMs to reliably output parseable JSON. The difference between a prototype and production.

## What you learn

- Why vague "return JSON" instructions fail inconsistently
- How to specify an exact schema so the model knows exactly what to output
- How to extract JSON from responses that include explanation text
- Parse recovery strategies when the model still wraps output in prose

## Run

```bash
python structured.py
```

## The problem

Without tight instruction: model might return
```
Here's the extracted information:
```json
{"name": "John"}
```
This contains the name extracted from the text.
```

With precise schema instruction: model returns
```json
{"name": "John", "age": 34, "sentiment": "POSITIVE"}
```

## Schema-first approach

1. Define the exact JSON schema in the system prompt
2. Instruct: "Respond ONLY with JSON. No other text."
3. Extract: strip markdown code fences, find `{...}` block
4. Validate: try parsing, check required keys
5. Retry on failure (with the parse error as feedback)
