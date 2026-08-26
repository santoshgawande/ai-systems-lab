from __future__ import annotations
"""
Async queue pattern for high-volume LLM workloads.
Demonstrates semaphore concurrency control, priority queue, retry with backoff.
Uses asyncio + httpx. No API key needed (Ollama).
"""
import os
import asyncio
import time
import random
import heapq
from dataclasses import dataclass, field
from typing import Any

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
MODEL = "phi4"
MAX_CONCURRENT = 4   # max parallel LLM calls


# ─── Job definition ───────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    priority: int          # lower = higher priority (heap ordering)
    prompt: str
    system: str = ""
    result: str | None = None
    error: str | None = None
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    def __lt__(self, other):
        return self.priority < other.priority

    @property
    def wait_ms(self) -> float | None:
        if self.started_at:
            return (self.started_at - self.queued_at) * 1000
        return None

    @property
    def process_ms(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


# ─── LLM caller ──────────────────────────────────────────────────────────────

async def call_ollama(prompt: str, system: str = "", max_retries: int = 3) -> str:
    import httpx
    for attempt in range(max_retries):
        try:
            payload = {"model": MODEL, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
                return r.json().get("response", "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(wait)
    return ""


# ─── Priority queue worker pool ──────────────────────────────────────────────

class LLMQueue:
    def __init__(self, max_workers: int = 4):
        self._heap: list[tuple[int, int, Job]] = []   # (priority, seq, job)
        self._seq = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._semaphore = asyncio.Semaphore(max_workers)
        self._results: dict[str, Job] = {}
        self._done_event: dict[str, asyncio.Event] = {}

    async def enqueue(self, job: Job):
        async with self._lock:
            heapq.heappush(self._heap, (job.priority, self._seq, job))
            self._seq += 1
            self._done_event[job.id] = asyncio.Event()
        self._not_empty.set()

    async def _dequeue(self) -> Job:
        while True:
            async with self._lock:
                if self._heap:
                    _, _, job = heapq.heappop(self._heap)
                    if not self._heap:
                        self._not_empty.clear()
                    return job
            await self._not_empty.wait()

    async def _process(self, job: Job):
        async with self._semaphore:
            job.started_at = time.time()
            try:
                job.result = await call_ollama(job.prompt, job.system)
            except Exception as e:
                job.error = str(e)
            job.completed_at = time.time()
            self._results[job.id] = job
            self._done_event[job.id].set()

    async def run_worker(self, n_jobs: int):
        processed = 0
        while processed < n_jobs:
            job = await self._dequeue()
            asyncio.create_task(self._process(job))
            processed += 1

    async def wait_for(self, job_id: str) -> Job:
        await self._done_event[job_id].wait()
        return self._results[job_id]


# ─── Demo ─────────────────────────────────────────────────────────────────────

JOBS_DATA = [
    (1, "critical-001", "Summarize in one sentence: machine learning is the process of..."),
    (1, "critical-002", "What is 2 + 2? Answer with just the number."),
    (2, "normal-001",   "List three benefits of containerization."),
    (2, "normal-002",   "What is a REST API? One sentence."),
    (2, "normal-003",   "What does ACID mean in databases?"),
    (3, "low-001",      "Write a haiku about Python programming."),
    (3, "low-002",      "What color is the sky? One word."),
    (3, "low-003",      "Name three programming languages."),
]


async def main():
    print("=== ASYNC LLM QUEUE DEMO ===\n")

    try:
        import httpx
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        print("Ollama not running. Showing queue mechanics.\n")
        print("Queue pattern pseudocode:")
        print("""
# Priority queue: lower number = higher priority
queue = PriorityQueue()
queue.put((1, Job("CRITICAL: urgent request")))
queue.put((3, Job("LOW: background task")))
queue.put((2, Job("NORMAL: regular request")))

# Jobs dequeue in priority order: 1, 2, 3
# even if enqueued out of order

# Semaphore: max N concurrent LLM calls
semaphore = asyncio.Semaphore(4)
async with semaphore:
    result = await llm_call(job.prompt)
    # at most 4 calls running simultaneously
""")
        print("Priority levels:")
        for priority, job_id, prompt in JOBS_DATA:
            print(f"  P{priority} [{job_id}]: {prompt[:50]!r}")
        return

    # Build jobs
    jobs = [
        Job(id=job_id, priority=priority, prompt=prompt)
        for priority, job_id, prompt in JOBS_DATA
    ]

    q = LLMQueue(max_workers=MAX_CONCURRENT)

    # Enqueue all jobs (mix of priorities)
    print(f"Enqueuing {len(jobs)} jobs (P1=critical, P2=normal, P3=low)...")
    for job in jobs:
        await q.enqueue(job)
        print(f"  Enqueued P{job.priority} [{job.id}]")
    print()

    # Start worker and wait for all results
    t_start = time.time()
    print(f"Processing with {MAX_CONCURRENT} concurrent workers...")
    worker = asyncio.create_task(q.run_worker(len(jobs)))

    results = await asyncio.gather(*[q.wait_for(j.id) for j in jobs])
    await worker

    elapsed = time.time() - t_start

    # Print results in priority order
    results_sorted = sorted(results, key=lambda j: (j.priority, j.id))
    print(f"\nResults (processed in {elapsed:.1f}s total):\n")
    print(f"  {'ID':<15} {'P'} {'Wait':>8} {'Process':>10}  Response")
    print(f"  {'-'*70}")
    for r in results_sorted:
        wait = f"{r.wait_ms:.0f}ms" if r.wait_ms is not None else "?"
        proc = f"{r.process_ms:.0f}ms" if r.process_ms is not None else "?"
        resp = (r.result or r.error or "?")[:40]
        print(f"  {r.id:<15} P{r.priority} {wait:>8} {proc:>10}  {resp!r}")

    # Stats
    completed = [r for r in results if r.result is not None]
    failed = [r for r in results if r.error is not None]
    avg_wait = sum(r.wait_ms for r in completed if r.wait_ms) / len(completed)
    avg_proc = sum(r.process_ms for r in completed if r.process_ms) / len(completed)

    print(f"\nStats:")
    print(f"  Completed: {len(completed)}/{len(jobs)}  Failed: {len(failed)}")
    print(f"  Avg wait:  {avg_wait:.0f}ms")
    print(f"  Avg proc:  {avg_proc:.0f}ms")
    print(f"  Throughput:{len(completed)/elapsed:.1f} req/s  (with {MAX_CONCURRENT} workers)")


if __name__ == "__main__":
    asyncio.run(main())
