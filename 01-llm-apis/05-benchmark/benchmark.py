"""
Lab 05 — Benchmark

Measures latency and throughput for each local model on the same prompt.
Runs N trials per model, reports p50 / p95 / tokens-per-second.

This is what you do before picking a model for a production feature.

Run:
  python benchmark.py                          # default models + prompt
  python benchmark.py --models phi4 llama3.3:70b --trials 3
  python benchmark.py --prompt "Summarise the history of the internet in 5 bullet points"
"""

import argparse
import statistics
import time

import requests
from rich.console import Console
from rich.table import Table

OLLAMA_URL = "http://localhost:11434"
console = Console()

DEFAULT_MODELS = ["phi4", "llama3.3:70b", "deepseek-r1"]
DEFAULT_PROMPT = "Explain what a vector database is and when you would use one. Be concise."
DEFAULT_TRIALS = 3


def single_call(model: str, prompt: str) -> dict:
    """One non-streaming call. Returns latency + token counts."""
    start = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.perf_counter() - start
    return {
        "latency_s": latency,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }


def benchmark_model(model: str, prompt: str, trials: int) -> dict:
    latencies = []
    output_tokens_list = []

    for i in range(1, trials + 1):
        console.print(f"  [{i}/{trials}] {model} ...", end=" ")
        try:
            result = single_call(model, prompt)
            latencies.append(result["latency_s"])
            output_tokens_list.append(result["output_tokens"])
            console.print(f"[green]{result['latency_s'] * 1000:.0f} ms[/green]")
        except Exception as exc:
            console.print(f"[red]ERROR: {exc}[/red]")

    if not latencies:
        return {"model": model, "error": "all trials failed"}

    avg_tokens = statistics.mean(output_tokens_list) if output_tokens_list else 0
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    avg_tps = avg_tokens / statistics.mean(latencies) if latencies else 0

    return {
        "model": model,
        "trials": len(latencies),
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "avg_output_tokens": avg_tokens,
        "tokens_per_sec": avg_tps,
        "error": None,
    }


def display_results(results: list[dict], prompt: str):
    console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")

    table = Table(title="Benchmark Results", header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("p50 latency", justify="right")
    table.add_column("p95 latency", justify="right")
    table.add_column("Avg output tokens", justify="right")
    table.add_column("Tokens / sec", justify="right")
    table.add_column("Status")

    for r in sorted(results, key=lambda x: x.get("p50_ms", float("inf"))):
        if r.get("error") and r["error"] == "all trials failed":
            table.add_row(r["model"], "—", "—", "—", "—", "[red]failed[/red]")
        else:
            table.add_row(
                r["model"],
                f"{r['p50_ms']:.0f} ms",
                f"{r['p95_ms']:.0f} ms",
                f"{r['avg_output_tokens']:.0f}",
                f"{r['tokens_per_sec']:.1f}",
                f"[green]{r['trials']} trials[/green]",
            )

    console.print(table)

    # verdict
    ok = [r for r in results if not r.get("error")]
    if ok:
        fastest = min(ok, key=lambda r: r["p50_ms"])
        best_tps = max(ok, key=lambda r: r["tokens_per_sec"])
        console.print(f"\n[bold]Fastest (p50):[/bold]   [cyan]{fastest['model']}[/cyan]  {fastest['p50_ms']:.0f} ms")
        console.print(f"[bold]Best throughput:[/bold] [cyan]{best_tps['model']}[/cyan]  {best_tps['tokens_per_sec']:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()

    console.print(f"[bold]Benchmarking {len(args.models)} models, {args.trials} trials each[/bold]\n")

    results = []
    for model in args.models:
        console.print(f"[bold]{model}[/bold]")
        result = benchmark_model(model, args.prompt, args.trials)
        results.append(result)
        console.print()

    display_results(results, args.prompt)


if __name__ == "__main__":
    main()
