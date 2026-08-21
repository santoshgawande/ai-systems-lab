"""
LiteLLM Router: load balancing, fallbacks, and retry across providers.
Handles rate limits, provider outages, and cost optimisation automatically.
"""
import os
import time

try:
    from litellm import Router
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("litellm not installed. pip install litellm\n")

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

MESSAGES = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]


def demo_concepts() -> None:
    print("""
LiteLLM Router concepts:

1. MODEL LIST — define all models with names, keys, rate limits:

from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-4o",           # logical name your code uses
            "litellm_params": {
                "model": "gpt-4o",            # actual LiteLLM model string
                "api_key": os.environ["OPENAI_API_KEY"],
                "rpm": 500,                   # rate limit: requests per minute
                "tpm": 150000,                # token limit: tokens per minute
            }
        },
        {
            "model_name": "gpt-4o",           # SAME logical name = load balancing
            "litellm_params": {
                "model": "azure/gpt-4o-prod",
                "api_base": "https://my.openai.azure.com",
                "api_key": os.environ["AZURE_API_KEY"],
                "rpm": 1000,
            }
        },
    ],
    routing_strategy="least-busy",            # or "latency-based", "usage-based-routing"
    fallbacks=[{"gpt-4o": ["claude-haiku-4-5-20251001"]}],  # fallback on failure
    num_retries=3,
    timeout=30,
    retry_after=5,                            # seconds between retries
)

# Use exactly like litellm.completion:
resp = router.completion(model="gpt-4o", messages=[...])

2. ROUTING STRATEGIES:
   "simple-shuffle"          round-robin across model_list entries
   "least-busy"              pick model with fewest in-flight requests
   "latency-based-routing"   pick model with lowest p95 latency
   "usage-based-routing"     pick model with most remaining budget

3. FALLBACKS:
   fallbacks=[{"gpt-4o": ["claude-opus-4-7", "ollama/llama3.2"]}]
   If gpt-4o fails → try claude → try ollama (in order)

4. COOLDOWNS:
   Router automatically puts models in cooldown after failures:
   - 429 (rate limit) → cooldown for retry_after seconds
   - 5xx errors → exponential backoff
   - timeout → mark model degraded

5. COST BUDGETS:
   router = Router(
       model_list=[...],
       budget_manager=litellm.BudgetManager(project_name="myapp"),
   )
   # Stops routing to models that exceed budget

6. ASYNC support:
   resp = await router.acompletion(model="gpt-4o", messages=[...])
   # Runs all fallback/retry logic async
""")


def run_router_demo() -> None:
    model_list = []
    fallback_chain = []

    if OPENAI_KEY:
        model_list.append({
            "model_name": "fast",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": OPENAI_KEY,
                "rpm": 500,
            }
        })
        fallback_chain.append("gpt-4o-mini")

    if ANTHROPIC_KEY:
        model_list.append({
            "model_name": "fast",
            "litellm_params": {
                "model": "claude-haiku-4-5-20251001",
                "api_key": ANTHROPIC_KEY,
                "rpm": 500,
            }
        })
        if "claude-haiku-4-5-20251001" not in fallback_chain:
            fallback_chain.append("claude-haiku-4-5-20251001")

    # Always add Ollama as a final local fallback
    model_list.append({
        "model_name": "fast",
        "litellm_params": {
            "model": "ollama/llama3.2",
            "api_base": OLLAMA_BASE,
        }
    })
    model_list.append({
        "model_name": "local",
        "litellm_params": {
            "model": "ollama/llama3.2",
            "api_base": OLLAMA_BASE,
        }
    })

    if not model_list:
        print("No providers configured.")
        return

    print(f"Configured {len(model_list)} model entries under logical name 'fast'")
    print(f"Fallback order: {fallback_chain + ['ollama/llama3.2']}\n")

    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        num_retries=2,
        timeout=15,
        fallbacks=[{"fast": ["local"]}],
    )

    # Batch of requests — router distributes them
    questions = [
        "What is 2+2?",
        "Capital of France?",
        "Name one planet in our solar system.",
    ]

    print("Sending 3 requests through router...\n")
    for q in questions:
        start = time.perf_counter()
        try:
            resp = router.completion(
                model="fast",
                messages=[{"role": "user", "content": q}],
                max_tokens=20,
            )
            elapsed = time.perf_counter() - start
            answer = resp.choices[0].message.content.strip()
            model_used = resp.model
            print(f"  Q: {q}")
            print(f"  A: {answer}  [{model_used}, {elapsed*1000:.0f}ms]")
        except Exception as e:
            print(f"  Q: {q}")
            print(f"  Error: {e}")
        print()

    # Show router stats
    try:
        stats = router.get_model_list()
        print("Router model pool:")
        for m in stats:
            name = m.get("model_name", "?")
            params = m.get("litellm_params", {})
            print(f"  {name}: {params.get('model', '?')}")
    except Exception:
        pass


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== LITELLM ROUTER DEMO ===\n")

    if not LITELLM_AVAILABLE:
        print("Install: pip install litellm")
        raise SystemExit(0)

    demo_concepts()
    run_router_demo()
