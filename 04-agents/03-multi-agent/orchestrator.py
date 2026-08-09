import sys
import json
import httpx

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"

PLANNER_SYSTEM = """You are a task planner. Break the user's goal into 3-5 concrete steps.

Respond ONLY with a JSON array. Each step:
{"step": 1, "description": "what to do", "expected_output": "what this produces"}

Example:
[
  {"step": 1, "description": "Define what RAG is", "expected_output": "A clear definition"},
  {"step": 2, "description": "List RAG advantages", "expected_output": "Bullet list of pros"}
]

Respond with valid JSON only. No markdown fences."""

EXECUTOR_SYSTEM = """You are a task executor. Execute the specific step given to you thoroughly.
Use results from previous steps as context. Be concise but complete."""

SYNTH_SYSTEM = """Combine multiple research steps into one coherent, well-structured final answer.
Use headers where appropriate. Be clear and direct."""


def call(system: str, user: str) -> str:
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def plan(goal: str) -> list[dict]:
    print(f"[Planner] Breaking down: {goal!r}")
    response = call(PLANNER_SYSTEM, goal)
    # Extract JSON array from response
    start, end = response.find("["), response.rfind("]") + 1
    if start == -1:
        raise ValueError(f"Planner returned no JSON:\n{response}")
    steps = json.loads(response[start:end])
    print(f"[Planner] {len(steps)} steps planned\n")
    return steps


def execute(step: dict, previous: list[str]) -> str:
    context = ""
    if previous:
        context = "\n\nPrevious results:\n" + "\n---\n".join(
            f"Step {i+1}: {r[:300]}" for i, r in enumerate(previous)
        )
    prompt = f"Execute: {step['description']}\nExpected output: {step['expected_output']}{context}"
    print(f"[Executor] Step {step['step']}: {step['description']}")
    result = call(EXECUTOR_SYSTEM, prompt)
    print(f"[Executor] Done ({len(result)} chars)\n")
    return result


def synthesize(goal: str, results: list[str]) -> str:
    prompt = f"Goal: {goal}\n\n" + "\n\n".join(
        f"Step {i+1} result:\n{r}" for i, r in enumerate(results)
    )
    return call(SYNTH_SYSTEM, prompt)


def run(goal: str):
    print(f"Goal: {goal}\n{'='*60}\n")
    steps = plan(goal)
    results = []
    for step in steps:
        results.append(execute(step, results))

    print("[Synthesizer] Combining all results...\n")
    final = synthesize(goal, results)
    print("=" * 60)
    print("Final Answer:\n")
    print(final)


goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
    "Explain the key differences between RAG and fine-tuning for production AI"
run(goal)
