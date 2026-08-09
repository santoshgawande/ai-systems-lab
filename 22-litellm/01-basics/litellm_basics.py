"""
LiteLLM: one API for every LLM provider.
Demonstrates provider switching, streaming, cost tracking, and async calls.
"""
import os
import asyncio

try:
    import litellm
    from litellm import completion, acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("litellm not installed. pip install litellm\n")

# ─── Config ───────────────────────────────────────────────────────────────────

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "ollama/llama3.2",
}

MESSAGES = [{"role": "user", "content": "In one sentence: what is a transformer neural network?"}]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def call_model(model: str, messages: list, provider_label: str) -> None:
    print(f"\n[{provider_label}] model={model}")
    try:
        resp = completion(
            model=model,
            messages=messages,
            max_tokens=100,
        )
        text = resp.choices[0].message.content
        usage = resp.usage
        cost = litellm.completion_cost(resp)
        print(f"  Response: {text}")
        print(f"  Tokens:   in={usage.prompt_tokens} out={usage.completion_tokens}")
        print(f"  Cost:     ${cost:.6f}")
    except Exception as e:
        print(f"  Error: {e}")


def streaming_demo(model: str) -> None:
    print(f"\n[Streaming] model={model}")
    print("  Response: ", end="", flush=True)
    try:
        stream = completion(
            model=model,
            messages=MESSAGES,
            max_tokens=80,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
        print()
    except Exception as e:
        print(f"\n  Error: {e}")


async def async_demo(models: list[tuple[str, str]]) -> None:
    print("\n[Async] calling multiple models concurrently...")
    async def call(model: str, label: str):
        try:
            resp = await acompletion(model=model, messages=MESSAGES, max_tokens=60)
            return label, resp.choices[0].message.content
        except Exception as e:
            return label, f"Error: {e}"

    tasks = [call(model, label) for label, model in models]
    results = await asyncio.gather(*tasks)
    for label, text in results:
        print(f"  [{label}]: {text}")


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== LITELLM BASICS DEMO ===\n")

if not LITELLM_AVAILABLE:
    print("""Install: pip install litellm

Usage pattern (same call for every provider):

import litellm

# OpenAI
resp = litellm.completion(model="gpt-4o-mini", messages=[...])

# Anthropic
resp = litellm.completion(model="claude-haiku-4-5-20251001", messages=[...])

# Gemini
resp = litellm.completion(model="gemini/gemini-2.0-flash", messages=[...])

# Ollama (local)
resp = litellm.completion(model="ollama/llama3.2", messages=[...],
                          api_base="http://localhost:11434")

# Cost tracking (automatic for cloud models)
cost = litellm.completion_cost(resp)
print(f"${cost:.6f}")

# Streaming (same flag for all providers)
stream = litellm.completion(model="gpt-4o-mini", messages=[...], stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Async (add 'a' prefix)
resp = await litellm.acompletion(model="gpt-4o-mini", messages=[...])
""")
    raise SystemExit(0)

# Enable verbose cost logging
litellm.success_callback = []
litellm.set_verbose = False

# Try each available provider
if OPENAI_KEY:
    call_model(MODELS["openai"], MESSAGES, "OpenAI")
else:
    print("\n[OpenAI] skipped — OPENAI_API_KEY not set")

if ANTHROPIC_KEY:
    call_model(MODELS["anthropic"], MESSAGES, "Anthropic")
else:
    print("\n[Anthropic] skipped — ANTHROPIC_API_KEY not set")

# Ollama — always try (local, no key needed)
ollama_model = MODELS["ollama"]
print(f"\n[Ollama] model={ollama_model}")
try:
    resp = completion(
        model=ollama_model,
        messages=MESSAGES,
        max_tokens=100,
        api_base=OLLAMA_BASE,
    )
    print(f"  Response: {resp.choices[0].message.content}")
    print(f"  Tokens:   in={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")
except Exception as e:
    print(f"  Error: {e}")
    print("  (Start Ollama and pull llama3.2: ollama pull llama3.2)")

# Streaming demo with whatever key is available
stream_model = None
if OPENAI_KEY:
    stream_model = MODELS["openai"]
elif ANTHROPIC_KEY:
    stream_model = MODELS["anthropic"]

if stream_model:
    streaming_demo(stream_model)

# Async concurrent demo
async_models = []
if OPENAI_KEY:
    async_models.append(("openai", MODELS["openai"]))
if ANTHROPIC_KEY:
    async_models.append(("anthropic", MODELS["anthropic"]))

if len(async_models) >= 2:
    asyncio.run(async_demo(async_models))

# Provider string reference
print("\n─── Provider model strings ───")
providers = [
    ("OpenAI",    "gpt-4o, gpt-4o-mini, gpt-3.5-turbo"),
    ("Anthropic", "claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5-20251001"),
    ("Gemini",    "gemini/gemini-2.0-flash, gemini/gemini-1.5-pro"),
    ("Ollama",    "ollama/llama3.2, ollama/mistral, ollama/phi3"),
    ("Azure",     "azure/<your-deployment-name>"),
    ("Cohere",    "command-r-plus, command-r"),
    ("Bedrock",   "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"),
]
for name, models_str in providers:
    print(f"  {name:<12} {models_str}")
