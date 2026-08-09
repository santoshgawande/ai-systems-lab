import hashlib
import httpx
from dataclasses import dataclass

OLLAMA = "http://localhost:11434"

@dataclass
class ModelSpec:
    name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    tier: str

MODELS = {
    "phi4":         ModelSpec("phi4",         0.0001, 0.0002, "cheap"),
    "llama3.2":     ModelSpec("llama3.2",     0.0003, 0.0006, "balanced"),
    "llama3.3:70b": ModelSpec("llama3.3:70b", 0.0012, 0.0018, "premium"),
}

CACHE: dict[str, str] = {}


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def cost_usd(model_name: str, input_t: int, output_t: int) -> float:
    m = MODELS.get(model_name)
    if not m:
        return 0.0
    return (input_t / 1000 * m.cost_per_1k_input) + (output_t / 1000 * m.cost_per_1k_output)


def classify_tier(prompt: str) -> str:
    low = prompt.lower()
    simple = ["what is", "define", "list", "yes or no", "convert", "translate", "format"]
    hard = ["analyze", "design", "architecture", "compare", "debug", "explain why", "trade-off"]
    if len(prompt) < 150 and any(s in low for s in simple):
        return "cheap"
    if any(s in low for s in hard):
        return "premium"
    return "balanced"


def route(tier: str) -> str:
    return {"cheap": "phi4", "balanced": "llama3.2", "premium": "llama3.3:70b"}.get(tier, "phi4")


def call_cached(prompt: str, model: str) -> tuple[str, bool]:
    key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
    if key in CACHE:
        return CACHE[key], True
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=60)
    r.raise_for_status()
    result = r.json()["message"]["content"]
    CACHE[key] = result
    return result, False


TASKS = [
    "What is the capital of France?",
    "What is the capital of France?",          # cache hit
    "Convert 100 Fahrenheit to Celsius.",
    "Analyze the trade-offs between RAG and fine-tuning for a production chatbot.",
    "Design a rate limiting system for an LLM API handling 10k requests/sec.",
    "List the days of the week.",
]

naive_model = "llama3.3:70b"
total_cost, naive_total = 0.0, 0.0

print(f"{'Task':<55} {'Tier':<10} {'Model':<16} {'Cache':<7} {'Cost'}")
print("-" * 105)

for task in TASKS:
    tier = classify_tier(task)
    model = route(tier)
    result, cached = call_cached(task, model)

    in_t = token_estimate(task)
    out_t = token_estimate(result)
    actual_cost = 0.0 if cached else cost_usd(model, in_t, out_t)
    naive_cost = cost_usd(naive_model, in_t, token_estimate("placeholder response length"))

    total_cost += actual_cost
    naive_total += naive_cost

    print(f"  {task[:52]:<52}  {tier:<10} {model:<16} {'HIT' if cached else 'miss':<7} ${actual_cost:.6f}")

print("-" * 105)
print(f"\nRouted + cached total:  ${total_cost:.6f}")
print(f"Naive (always premium): ${naive_total:.6f}")
if naive_total > 0:
    savings_pct = (1 - total_cost / naive_total) * 100
    print(f"Savings:                {savings_pct:.0f}%")
