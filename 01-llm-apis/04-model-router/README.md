# Lab 04 — Model Router

Classify a prompt by task type and route it to the best model. The same pattern used inside LiteLLM, Portkey, and OpenRouter.

## What you learn

- Intent classification: code / reasoning / fast / general
- Routing table: map task types to specialist models
- Cost optimisation: don't send every request to the big expensive model

## Run

```bash
pip install requests
python router.py "Write a function to parse JSON in Python"
python router.py "If all roses are flowers, do roses need water?"
python router.py "What is the capital of France?"
python router.py --list-models
```

## Routing rules

| Task type | Model | Why |
|-----------|-------|-----|
| `code` | `qwen2.5-coder:32b` | Trained on code, better completions |
| `reasoning` | `deepseek-r1` | Step-by-step chain-of-thought |
| `fast` | `phi4` | Low latency for simple queries |
| `general` | `llama3.3:70b` | Balanced default |

## Pattern

```python
def route(prompt: str) -> str:
    task = classify(prompt)   # code / reasoning / fast / general
    return MODEL_MAP[task]

def classify(prompt: str) -> str:
    # Fast heuristic: keyword matching
    if any(w in prompt for w in ["function", "code", "python", "sql"]):
        return "code"
    ...
    # Or ask a cheap model to classify
    return llm(f"Classify this prompt: {prompt}\nReturn: code/reasoning/fast/general")
```

## Key insight

Routing to the right model cuts cost by 5-10× on mixed workloads without sacrificing quality.
Most requests are simple — only a few need the big model.
