"""
Lab 02 — Reading real token counts from Ollama's API

Ollama returns token usage in every response:
  prompt_eval_count  = input tokens (what the model read)
  eval_count         = output tokens (what the model generated)
  prompt_eval_duration / eval_duration = time in nanoseconds

This lets you verify your tokenizer estimates against what the model actually saw.
"""

import json
import os
import requests
from rich.console import Console
from rich.table import Table
from token_utils import count_tokens, count_tokens_in_messages

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

console = Console()


def chat(messages: list[dict], model: str = MODEL) -> dict:
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": messages,
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_scenario(label: str, messages: list[dict]) -> dict:
    console.print(f"\n[bold]Running:[/bold] {label}")
    try:
        result = chat(messages)
    except Exception as e:
        console.print(f"[red]Ollama error: {e}[/red]")
        console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
        return {}

    actual_input = result.get("prompt_eval_count", 0)
    actual_output = result.get("eval_count", 0)
    estimated_input = count_tokens_in_messages(messages)

    return {
        "label": label,
        "actual_input": actual_input,
        "estimated_input": estimated_input,
        "estimate_error_pct": abs(actual_input - estimated_input) / max(actual_input, 1) * 100,
        "actual_output": actual_output,
        "response": result.get("message", {}).get("content", "")[:80],
    }


def main():
    console.rule("[bold blue]Lab 02 — Real token counts from Ollama")

    scenarios = [
        (
            "Short question",
            [{"role": "user", "content": "What is 2 + 2?"}],
        ),
        (
            "Long system prompt + question",
            [
                {"role": "system", "content": "You are a helpful assistant specializing in "
                 "mathematics, science, history, geography, and general knowledge. "
                 "Always provide detailed, accurate, and well-structured answers. "
                 "Use bullet points and numbered lists where appropriate."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        ),
        (
            "Multi-turn conversation",
            [
                {"role": "user", "content": "My name is Santosh."},
                {"role": "assistant", "content": "Hello Santosh! How can I help you today?"},
                {"role": "user", "content": "What is my name?"},
            ],
        ),
        (
            "JSON-heavy context (like tool results)",
            [
                {"role": "user", "content": "Summarize this data:"},
                {"role": "user", "content": json.dumps({
                    "weather": {"city": "San Francisco", "temp": 18, "humidity": 72,
                                "wind_speed": 15, "condition": "Partly cloudy"},
                    "forecast": [{"day": "Mon", "high": 19, "low": 12},
                                 {"day": "Tue", "high": 21, "low": 13},
                                 {"day": "Wed", "high": 17, "low": 11}]
                }, indent=2)},
            ],
        ),
    ]

    results = [run_scenario(label, msgs) for label, msgs in scenarios]
    results = [r for r in results if r]

    if not results:
        return

    console.rule("[bold blue]Results")
    table = Table(show_header=True)
    table.add_column("Scenario", max_width=30)
    table.add_column("Actual Input", justify="right")
    table.add_column("Est. Input", justify="right")
    table.add_column("Err %", justify="right")
    table.add_column("Output", justify="right")

    for r in results:
        err_color = "green" if r["estimate_error_pct"] < 15 else "yellow"
        table.add_row(
            r["label"],
            str(r["actual_input"]),
            str(r["estimated_input"]),
            f"[{err_color}]{r['estimate_error_pct']:.1f}%[/{err_color}]",
            str(r["actual_output"]),
        )

    console.print(table)
    console.print("\n[yellow]Key insight:[/yellow] tiktoken estimate is ~5-15% off for Llama models.")
    console.print("In production, always use the model's own tokenizer for accurate billing.")


if __name__ == "__main__":
    main()
