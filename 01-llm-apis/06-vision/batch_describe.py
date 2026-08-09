"""
Lab 06d — Batch Image Processing

Describe or analyse all images in a folder using a vision model.
Results are printed to console and saved to a JSON file.

Run:
  python batch_describe.py samples/                                      # describe all images
  python batch_describe.py samples/ "What objects are in this image?"    # custom question
  python batch_describe.py samples/ --model moondream --workers 2        # faster with moondream
  python batch_describe.py samples/ --out results.json                   # save results

Use cases:
  - Auto-caption a photo library
  - Audit screenshots for UI elements
  - Tag images with detected content
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "moondream"  # fastest for batch work
DEFAULT_QUESTION = "Describe what you see in this image in 2-3 sentences."
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
console = Console()


def find_images(folder: str) -> list[Path]:
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def describe_image(model: str, image_path: Path, question: str) -> dict:
    start = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

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
            "file": image_path.name,
            "path": str(image_path),
            "description": data["message"]["content"].strip(),
            "output_tokens": data.get("eval_count", 0),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "error": None,
        }
    except Exception as exc:
        return {
            "file": image_path.name,
            "path": str(image_path),
            "description": "",
            "output_tokens": 0,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch-describe images in a folder")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to ask about each image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    parser.add_argument("--out", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    images = find_images(args.folder)
    if not images:
        console.print(f"[red]No images found in {args.folder}[/red]")
        console.print(f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    console.print(f"\n[bold]Folder  :[/bold] {args.folder}")
    console.print(f"[bold]Images  :[/bold] {len(images)}")
    console.print(f"[bold]Model   :[/bold] {args.model}")
    console.print(f"[bold]Workers :[/bold] {args.workers}")
    console.print(f"[bold]Question:[/bold] {args.question}\n")

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(images)} images...", total=len(images))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(describe_image, args.model, img, args.question): img for img in images}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "[red]✗[/red]" if result["error"] else "[green]✓[/green]"
                progress.console.print(f"  {status} {result['file']}")
                progress.advance(task)

    results.sort(key=lambda r: r["file"])

    # print results
    console.print("\n" + "─" * 60)
    for r in results:
        if r["error"]:
            console.print(f"\n[bold red]{r['file']}[/bold red]")
            console.print(f"  [red]Error: {r['error']}[/red]")
        else:
            console.print(f"\n[bold cyan]{r['file']}[/bold cyan]  [dim]{r['latency_ms']:.0f} ms[/dim]")
            console.print(f"  {r['description']}")

    # summary table
    ok = [r for r in results if not r["error"]]
    if ok:
        avg_latency = sum(r["latency_ms"] for r in ok) / len(ok)
        total_tokens = sum(r["output_tokens"] for r in ok)

        table = Table(title="\nBatch Summary", header_style="bold")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_row("Images processed", str(len(ok)))
        table.add_row("Errors", str(len(results) - len(ok)))
        table.add_row("Avg latency", f"{avg_latency:.0f} ms")
        table.add_row("Total output tokens", str(total_tokens))
        console.print(table)

    if args.out:
        output = {
            "model": args.model,
            "question": args.question,
            "total_images": len(results),
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\n[green]Results saved to {args.out}[/green]")


if __name__ == "__main__":
    main()
