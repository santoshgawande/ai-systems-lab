"""
Lab 04 — Model Router

Classifies a prompt by task type and routes it to the best local model.
This is the same pattern used inside production AI gateways (LiteLLM, Portkey, OpenRouter).

Routing rules:
  code      → qwen2.5-coder:32b  (specialised code model)
  reasoning → deepseek-r1         (step-by-step reasoning)
  fast      → phi4                (lightweight, low latency)
  general   → llama3.3:70b        (default, balanced)

Run:
  python router.py "Write a function to parse JSON in Python"
  python router.py "If all roses are flowers and all flowers need water, do roses need water?"
  python router.py "What is the capital of France?"
  python router.py --list-models
"""

import argparse
import json
import time

import requests
from rich.console import Console

OLLAMA_URL = "http://localhost:11434"
console = Console()

# -------------------------------------------------------------------
# Routing table — edit model tags to match your `ollama list` output
# -------------------------------------------------------------------
ROUTING_TABLE = {
    "code": {
        "model": "qwen2.5-coder:32b",
        "description": "Code generation, debugging, refactoring",
        "keywords": ["write", "function", "code", "implement", "debug", "script",
                     "python", "javascript", "class", "algorithm", "program"],
    },
    "reasoning": {
        "model": "deepseek-r1",
        "description": "Math, logic, step-by-step reasoning",
        "keywords": ["solve", "prove", "if", "therefore", "logic", "math",
                     "calculate", "equation", "deduce", "why", "reason"],
    },
    "fast": {
        "model": "phi4",
        "description": "Short factual questions, classification",
        "keywords": ["what is", "who is", "define", "capital", "list", "name",
                     "yes or no", "true or false", "quick", "simple"],
    },
    "general": {
        "model": "llama3.3:70b",
        "description": "Default — balanced quality and speed",
        "keywords": [],  # fallback
    },
}


def classify_prompt(prompt: str) -> str:
    """Simple keyword-based classifier. Returns a task type key."""
    lower = prompt.lower()
    scores = {task: 0 for task in ROUTING_TABLE}

    for task, config in ROUTING_TABLE.items():
        for kw in config["keywords"]:
            if kw in lower:
                scores[task] += 1

    # exclude 'general' from keyword scoring — it's always the fallback
    scores.pop("general")
    best_task = max(scores, key=lambda t: scores[t]) if any(scores.values()) else "general"
    return best_task if scores.get(best_task, 0) > 0 else "general"


def call_model(model: str, prompt: str) -> dict:
    start = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.perf_counter() - start
    return {
        "answer": data["message"]["content"],
        "output_tokens": data.get("eval_count", 0),
        "latency_ms": elapsed * 1000,
    }


def route(prompt: str) -> None:
    task = classify_prompt(prompt)
    config = ROUTING_TABLE[task]
    model = config["model"]

    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print(f"[dim]Task type : [cyan]{task}[/cyan] — {config['description']}[/dim]")
    console.print(f"[dim]Routed to : [green]{model}[/green][/dim]\n")
    console.print("─" * 60)

    result = call_model(model, prompt)

    console.print(result["answer"].strip())
    console.print(f"\n[dim]Latency: {result['latency_ms']:.0f} ms · "
                  f"Output tokens: {result['output_tokens']}[/dim]")


def list_models() -> None:
    table_data = [
        (task, cfg["model"], cfg["description"], ", ".join(cfg["keywords"][:5]) or "—")
        for task, cfg in ROUTING_TABLE.items()
    ]
    console.print("\n[bold]Routing Table[/bold]\n")
    for task, model, desc, kws in table_data:
        console.print(f"  [cyan]{task:12s}[/cyan] → [green]{model:30s}[/green]  {desc}")
        if kws != "—":
            console.print(f"  {' ' * 14}  keywords: [dim]{kws}[/dim]")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Route a prompt to the best local model")
    parser.add_argument("prompt", nargs="?", help="Prompt to route and answer")
    parser.add_argument("--list-models", action="store_true", help="Show routing table")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.prompt:
        parser.print_help()
        return

    route(args.prompt)


if __name__ == "__main__":
    main()
