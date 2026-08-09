"""
Lab 06a — Image Q&A

Ask a vision model any question about an image.
Accepts a local file path or a URL (downloads and encodes automatically).

Run:
  python image_qa.py samples/photo.jpg "What is in this image?"
  python image_qa.py samples/chart.png "What trend does this chart show?"
  python image_qa.py samples/screenshot.png "What UI elements are visible?"
  python image_qa.py --model moondream samples/photo.jpg "Describe this scene"
"""

import argparse
import base64
import io
import sys
import time
import urllib.request

import requests
from rich.console import Console
from rich.panel import Panel

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llava"
console = Console()


def load_image_as_base64(source: str) -> tuple[str, str]:
    """Load image from file path or URL. Returns (base64_str, source_label)."""
    if source.startswith("http://") or source.startswith("https://"):
        console.print(f"[dim]Downloading image from URL...[/dim]")
        with urllib.request.urlopen(source) as response:
            image_bytes = response.read()
        return base64.b64encode(image_bytes).decode("utf-8"), source
    else:
        with open(source, "rb") as f:
            image_bytes = f.read()
        size_kb = len(image_bytes) / 1024
        return base64.b64encode(image_bytes).decode("utf-8"), f"{source} ({size_kb:.0f} KB)"


def ask_vision_model(model: str, image_b64: str, question: str) -> dict:
    start = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [image_b64],
                }
            ],
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


def main():
    parser = argparse.ArgumentParser(description="Ask a vision model about an image")
    parser.add_argument("image", help="Path to image file or URL")
    parser.add_argument("question", help="Question to ask about the image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Vision model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    console.print(f"\n[bold]Model   :[/bold] {args.model}")
    console.print(f"[bold]Question:[/bold] {args.question}")

    image_b64, label = load_image_as_base64(args.image)
    console.print(f"[bold]Image   :[/bold] {label}")
    console.print(f"[dim]Image size (base64): {len(image_b64) // 1024} KB[/dim]\n")
    console.print("─" * 60)

    result = ask_vision_model(args.model, image_b64, args.question)

    console.print(Panel(
        result["answer"].strip(),
        title=f"[bold cyan]{args.model}[/bold cyan]",
        border_style="cyan",
    ))
    console.print(f"[dim]Latency: {result['latency_ms']:.0f} ms · Output tokens: {result['output_tokens']}[/dim]")


if __name__ == "__main__":
    main()
