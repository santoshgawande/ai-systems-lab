import time
import json
import uuid
import httpx
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

OLLAMA = "http://localhost:11434"
LOG_FILE = "/tmp/llm_traces.jsonl"

@dataclass
class Trace:
    trace_id: str
    model: str
    prompt_preview: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float
    status: str
    error: Optional[str]
    timestamp: str


traces: list[Trace] = []


def call_traced(model: str, prompt: str) -> tuple[str, Trace]:
    trace_id = uuid.uuid4().hex[:8]
    start = time.time()
    ttft_ms = 0.0
    parts: list[str] = []
    input_tokens = output_tokens = 0
    status, error = "ok", None

    try:
        with httpx.stream("POST", f"{OLLAMA}/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }, timeout=60) as r:
            r.raise_for_status()
            first = True
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    if first:
                        ttft_ms = (time.time() - start) * 1000
                        first = False
                    parts.append(token)
                if chunk.get("done"):
                    input_tokens = chunk.get("prompt_eval_count", 0)
                    output_tokens = chunk.get("eval_count", 0)
    except httpx.TimeoutException as e:
        status, error = "timeout", str(e)
    except Exception as e:
        status, error = "error", str(e)

    latency_ms = (time.time() - start) * 1000
    trace = Trace(
        trace_id=trace_id,
        model=model,
        prompt_preview=prompt[:60].replace("\n", " "),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms, 1),
        ttft_ms=round(ttft_ms, 1),
        status=status,
        error=error,
        timestamp=datetime.now().isoformat(),
    )

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(asdict(trace)) + "\n")

    traces.append(trace)
    return "".join(parts), trace


CALLS = [
    ("phi4",    "What is 2 + 2?"),
    ("phi4",    "List 5 programming languages."),
    ("phi4",    "What is an embedding in machine learning?"),
    ("phi4",    "What is the capital of Japan?"),
    ("llama3.2","Compare REST and GraphQL APIs."),
]

print("Running traced calls...\n")
print(f"{'ID':<10} {'Model':<14} {'St':<4} {'In':>5} {'Out':>5}  {'TTFT':>7}  {'Total':>8}  Prompt")
print("-" * 95)

for model, prompt in CALLS:
    _, t = call_traced(model, prompt)
    icon = "✓" if t.status == "ok" else "✗"
    print(f"{t.trace_id:<10} {t.model:<14} {icon:<4} {t.input_tokens:>5} {t.output_tokens:>5}  {t.ttft_ms:>6.0f}ms  {t.latency_ms:>7.0f}ms  {t.prompt_preview[:35]!r}")

ok = [t for t in traces if t.status == "ok"]
print(f"\n{'='*95}")
if ok:
    lats = sorted(t.latency_ms for t in ok)
    p50 = lats[len(lats) // 2]
    p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)]
    ttfts = [t.ttft_ms for t in ok if t.ttft_ms > 0]

    print(f"Calls: {len(traces)}  Success: {len(ok)}  Errors: {len(traces)-len(ok)}")
    print(f"Total tokens: {sum(t.input_tokens+t.output_tokens for t in ok)} ({sum(t.input_tokens for t in ok)} in / {sum(t.output_tokens for t in ok)} out)")
    print(f"Latency p50/p95: {p50:.0f}ms / {p95:.0f}ms")
    if ttfts:
        print(f"TTFT avg: {sum(ttfts)/len(ttfts):.0f}ms")
    print(f"\nFull traces: {LOG_FILE}")
