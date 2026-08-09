import time
import random
import httpx
from dataclasses import dataclass

OLLAMA = "http://localhost:11434"

@dataclass
class Model:
    name: str
    priority: int  # lower = try first


MODELS = [
    Model("phi4", 1),
    Model("llama3.2", 2),
    Model("llama3.3:70b", 3),
]

_fail_count = 0  # used by simulated failures


class RateLimitError(Exception):
    pass


class LLMError(Exception):
    def __init__(self, msg: str, retryable: bool = True):
        super().__init__(msg)
        self.retryable = retryable


def simulate_failure():
    """First 2 calls timeout, 3rd is a rate limit, 4th succeeds."""
    global _fail_count
    _fail_count += 1
    if _fail_count <= 2:
        raise httpx.TimeoutException("Simulated timeout")
    if _fail_count == 3:
        raise RateLimitError("Simulated 429 rate limit")


def call(model: str, prompt: str, inject_failures: bool = False) -> str:
    if inject_failures:
        simulate_failure()
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=20)
    if r.status_code == 429:
        raise RateLimitError(f"Rate limited on {model}")
    if r.status_code >= 500:
        raise LLMError(f"Server error {r.status_code}", retryable=True)
    if r.status_code >= 400:
        raise LLMError(f"Client error {r.status_code}", retryable=False)
    r.raise_for_status()
    return r.json()["message"]["content"]


def with_retry(model: str, prompt: str, max_attempts: int = 4,
               base_delay: float = 0.5, inject: bool = False) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return call(model, prompt, inject_failures=inject)
        except (httpx.TimeoutException, RateLimitError) as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            print(f"    [retry {attempt}/{max_attempts}] {type(e).__name__}: sleeping {delay:.2f}s")
            time.sleep(delay)
        except LLMError as e:
            if not e.retryable or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            print(f"    [retry {attempt}/{max_attempts}] {e}: sleeping {delay:.2f}s")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


def with_fallback(prompt: str) -> tuple[str, str]:
    for model in sorted(MODELS, key=lambda m: m.priority):
        try:
            print(f"  [fallback] trying {model.name}")
            result = with_retry(model.name, prompt)
            return result, model.name
        except Exception as e:
            print(f"  [fallback] {model.name} failed: {e}")
    raise RuntimeError("All models failed")


prompt = "What is 2 + 2? One word answer."

print("Demo 1: Normal call with retry")
result = with_retry("phi4", prompt)
print(f"  Result: {result.strip()}\n")

print("Demo 2: Fallback routing (tries models in order)")
result, used = with_fallback(prompt)
print(f"  Result: {result.strip()}  (used: {used})\n")

print("Demo 3: Simulated failures → retry → success")
global _fail_count
_fail_count = 0
try:
    result = with_retry("phi4", prompt, max_attempts=5, inject=True)
    print(f"  Result: {result.strip()}\n")
except Exception as e:
    print(f"  Failed: {e}\n")
