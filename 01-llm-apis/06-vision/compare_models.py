"""
Lab 06b — Compare Vision Models

Same image + same question sent to multiple vision models in parallel.
Shows you quality vs speed trade-offs between models.

Run:
  python compare_models.py samples/photo.jpg "Describe this image in detail"
  python compare_models.py samples/chart.png "What does this chart show?" --models llava qwen2.5vl:7b
  python compare_models.py samples/invoice.jpg "List all line items and prices"
"""

import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODELS = ["llava", "moondream", "qwen2.5vl:7b"]
console = Console()


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vision_model(model: str, image_b64: str, question: str) -> dict:
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": question, "images": [image_b64]}],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "model": model,
            "answer": data["message"]["content"],
            "output_tokens": data.get("eval_count", 0),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model,
            "answer": "",
            "output_tokens": 0,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Compare vision models on the same image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("question", help="Question to ask about the image")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    console.print(f"\n[bold]Image   :[/bold] {args.image}")
    console.print(f"[bold]Question:[/bold] {args.question}")
    console.print(f"[dim]Models: {', '.join(args.models)} — querying in parallel...[/dim]\n")

    image_b64 = load_image_b64(args.image)

    results = []
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        futures = {pool.submit(query_vision_model, m, image_b64, args.question): m for m in args.models}
        for future in as_completed(futures):
            r = future.result()
            status = "[red]ERROR[/red]" if r["error"] else "[green]done[/green]"
            console.print(f"  {r['model']:25s} {status}  ({r['latency_ms']:.0f} ms)")
            results.append(r)

    results.sort(key=lambda r: r["latency_ms"])

    console.print()
    for r in results:
        if r["error"]:
            console.print(Panel(
                f"[red]{r['error']}[/red]",
                title=f"[bold red]{r['model']}[/bold red]",
                border_style="red",
            ))
        else:
            console.print(Panel(
                r["answer"].strip(),
                title=f"[bold cyan]{r['model']}[/bold cyan]  "
                      f"[dim]{r['latency_ms']:.0f} ms · {r['output_tokens']} tokens[/dim]",
                border_style="cyan",
            ))

    # summary table
    table = Table(title="Summary", header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("Status")

    for r in results:
        tps = f"{r['output_tokens'] / (r['latency_ms'] / 1000):.1f}" if r["latency_ms"] > 0 else "—"
        status = "[red]error[/red]" if r["error"] else "[green]ok[/green]"
        table.add_row(r["model"], f"{r['latency_ms']:.0f}", str(r["output_tokens"]), tps, status)

    console.print(table)


if __name__ == "__main__":
    main()
