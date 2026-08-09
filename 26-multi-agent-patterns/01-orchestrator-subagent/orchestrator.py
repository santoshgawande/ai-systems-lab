"""
Orchestrator-Subagent pattern: a planner LLM decomposes a task and dispatches
specialist subagents (researcher, writer, critic), then aggregates results.
"""
import os
import json
import asyncio
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# ─── Base LLM call ────────────────────────────────────────────────────────────

async def llm_async(prompt: str, system: str = "", max_tokens: int = 400) -> str:
    if OPENAI_KEY:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": max_tokens},
                timeout=30,
            )
            return r.json()["choices"][0]["message"]["content"].strip()

    if ANTHROPIC_KEY:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": max_tokens,
                    "system": system or "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            return r.json()["content"][0]["text"].strip()

    # Ollama
    full = f"{system}\n\n{prompt}".strip() if system else prompt
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": full, "stream": False},
            timeout=60,
        )
        return r.json()["response"].strip()


# ─── Subagents ────────────────────────────────────────────────────────────────

async def research_agent(topic: str) -> dict:
    """Gather key facts about a topic."""
    result = await llm_async(
        f"List 5 key facts about: {topic}\nFormat as a numbered list. Be concise.",
        system="You are a research specialist. Provide accurate, factual summaries.",
        max_tokens=250,
    )
    return {"agent": "researcher", "topic": topic, "output": result}


async def writer_agent(topic: str, facts: str) -> dict:
    """Write a short article based on research."""
    result = await llm_async(
        f"Write a 3-paragraph summary about '{topic}' using these facts:\n{facts}",
        system="You are a technical writer. Write clearly and concisely for a software engineering audience.",
        max_tokens=350,
    )
    return {"agent": "writer", "output": result}


async def critic_agent(article: str) -> dict:
    """Review the article for accuracy and clarity."""
    result = await llm_async(
        f"Review this article. Rate clarity (1-10) and accuracy (1-10). Give 2 specific improvement suggestions.\n\nArticle:\n{article}",
        system="You are a critical editor. Be constructive and specific.",
        max_tokens=200,
    )
    return {"agent": "critic", "output": result}


# ─── Orchestrator ─────────────────────────────────────────────────────────────

async def orchestrator(task: str) -> dict:
    """
    Decomposes task → dispatches subagents → aggregates results.
    Research runs first (blocking), then writer and critic run in parallel.
    """
    print(f"[Orchestrator] Task: {task!r}")
    print(f"[Orchestrator] Step 1: Research...")

    # Stage 1: research (blocking — writer needs the output)
    research = await research_agent(task)
    print(f"[Researcher]   Done. {len(research['output'])} chars")

    print(f"[Orchestrator] Step 2: Writing + critique in parallel...")

    # Stage 2: writer and an independent critic review in parallel
    writer_task = writer_agent(task, research["output"])
    # Critic reviews the research directly (doesn't need final article)
    critic_preview_task = critic_agent(research["output"])

    writer_result, critic_preview = await asyncio.gather(writer_task, critic_preview_task)
    print(f"[Writer]       Done. {len(writer_result['output'])} chars")
    print(f"[Critic]       Done.")

    print(f"[Orchestrator] Step 3: Final synthesis...")

    # Stage 3: orchestrator synthesises everything
    synthesis_prompt = f"""You orchestrated these agents for the task: "{task}"

Research findings:
{research['output']}

Written article:
{writer_result['output']}

Critic feedback:
{critic_preview['output']}

Write a final 2-sentence executive summary of the key insight and the main improvement needed."""

    final = await llm_async(synthesis_prompt, max_tokens=150)

    return {
        "task": task,
        "stages": [research, writer_result, critic_preview],
        "final_summary": final,
    }


# ─── Sequential for comparison ────────────────────────────────────────────────

async def sequential_pipeline(task: str) -> float:
    """Same work but sequential — to benchmark vs parallel orchestrator."""
    import time
    t0 = time.perf_counter()
    r = await research_agent(task)
    w = await writer_agent(task, r["output"])
    c = await critic_agent(w["output"])
    return time.perf_counter() - t0


# ─── Demo ─────────────────────────────────────────────────────────────────────

async def main():
    import time

    print("=== ORCHESTRATOR-SUBAGENT DEMO ===\n")

    # Check connectivity
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        ollama_ok = True
    except Exception:
        ollama_ok = False

    if not OPENAI_KEY and not ANTHROPIC_KEY and not ollama_ok:
        print("No LLM available. Start Ollama or set OPENAI_API_KEY / ANTHROPIC_API_KEY\n")
        print_concepts()
        return

    task = "Vector databases for AI applications"

    t0 = time.perf_counter()
    result = await orchestrator(task)
    elapsed = time.perf_counter() - t0

    print(f"\n{'─'*60}")
    print("RESULTS\n")

    for stage in result["stages"]:
        print(f"[{stage['agent'].upper()}]")
        print(f"{stage['output'][:300]}...")
        print()

    print(f"[FINAL SUMMARY]")
    print(result["final_summary"])
    print(f"\nTotal time (parallel): {elapsed:.1f}s")


def print_concepts():
    print("""
Orchestrator-Subagent pattern:

async def orchestrator(task):
    # Stage 1: sequential (dependencies)
    research = await research_agent(task)

    # Stage 2: parallel (independent)
    writer, critic = await asyncio.gather(
        writer_agent(task, research.output),
        critic_agent(research.output),
    )

    # Stage 3: synthesise
    return synthesise(research, writer, critic)

Key ideas:
  1. Map dependencies: what MUST run before what?
  2. Parallelise independent stages with asyncio.gather()
  3. Each subagent has its OWN system prompt and tools
  4. Orchestrator owns the plan; subagents own execution

When to use:
  - Task has clear decomposition into specialist subtasks
  - Subtasks can run in parallel (>1 independent stage)
  - Each stage's output feeds into the next (pipeline)
  - Context window would overflow in single-agent mode
""")


if __name__ == "__main__":
    asyncio.run(main())
