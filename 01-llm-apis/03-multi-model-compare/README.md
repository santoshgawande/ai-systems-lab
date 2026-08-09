# Lab 03 — Multi-Model Compare

Send the same prompt to multiple local models in parallel and compare quality + speed side by side.

## What you learn

- `ThreadPoolExecutor` for parallel API calls (avoids sequential N×latency)
- How to objectively compare model outputs for the same task
- Quality vs speed trade-offs across model families (Llama, Qwen, Phi, DeepSeek)

## Run

```bash
pip install requests rich
python compare.py "What is a transformer?"
python compare.py "Write a Python function to reverse a string" --models llama3.2 phi4 qwen2.5-coder:7b
python compare.py "Solve: 2x + 3 = 11" --models deepseek-r1 llama3.2
```

## Pattern

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def call_model(model):
    return model, chat(model, prompt)

with ThreadPoolExecutor() as pool:
    futures = {pool.submit(call_model, m): m for m in models}
    for f in as_completed(futures):
        model, result = f.result()
        print(model, result)
```

## What to compare

- **Factual accuracy** — does it get the answer right?
- **Format compliance** — does it follow the requested format?
- **Conciseness** — does it pad with unnecessary text?
- **Latency** — how long does it take to respond?

Running the same prompt across models is the fastest way to pick the right one for your task.
