# Lab 02 — LiteLLM Router

Load balancing, fallbacks, and retries across providers. Never go down because one provider is rate-limited.

## What you learn

- `Router` with multiple model entries under one logical name
- Routing strategies: `least-busy`, `latency-based-routing`, `usage-based-routing`
- Automatic fallback chain: gpt-4o → claude → ollama
- Cooldown on rate limits and 5xx errors

## Run

```bash
pip install litellm
export OPENAI_API_KEY=sk-...       # optional
export ANTHROPIC_API_KEY=sk-...    # optional
python routing.py
```

## Key pattern

```python
from litellm import Router

router = Router(
    model_list=[
        # Two entries with same model_name = load balanced
        {"model_name": "fast", "litellm_params": {"model": "gpt-4o-mini", ...}},
        {"model_name": "fast", "litellm_params": {"model": "claude-haiku-4-5-20251001", ...}},
        # Fallback target
        {"model_name": "local", "litellm_params": {"model": "ollama/llama3.2", "api_base": "..."}},
    ],
    routing_strategy="least-busy",
    fallbacks=[{"fast": ["local"]}],   # if "fast" fails, try "local"
    num_retries=3,
    timeout=30,
)

resp = router.completion(model="fast", messages=[...])
```

## When to use Router in production

- Multiple OpenAI keys to stay under per-key rate limits
- Azure OpenAI + OpenAI as hot standby
- Cloud LLM + local Ollama as cost fallback
- Gradual traffic migration to a new model
