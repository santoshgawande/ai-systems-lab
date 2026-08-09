"""
Lab 06c — OCR & Structured Extraction

Extract text and structured data from images using vision models.
Useful for invoices, screenshots, forms, charts, whiteboards.

Run:
  python ocr_extract.py samples/screenshot.png              # extract all visible text
  python ocr_extract.py samples/invoice.jpg --mode invoice  # structured invoice extraction
  python ocr_extract.py samples/chart.png --mode chart      # extract chart data as table
  python ocr_extract.py samples/form.png --mode form        # extract form fields + values

Modes:
  text     — extract all text verbatim (default)
  invoice  — extract vendor, date, line items, total as JSON
  chart    — describe axes, data series, trends
  form     — extract field labels and their values as JSON
  table    — extract any tabular data as CSV
"""

import argparse
import base64
import json

import requests
from rich.console import Console
from rich.syntax import Syntax

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"  # best for OCR and structured extraction
console = Console()

PROMPTS = {
    "text": (
        "Extract ALL text visible in this image exactly as it appears. "
        "Preserve formatting, line breaks, and layout as much as possible. "
        "Do not add any commentary — just the extracted text."
    ),
    "invoice": (
        "Extract the invoice data from this image and return it as valid JSON only. "
        "Schema: {vendor: string, date: string, invoice_number: string, "
        "line_items: [{description, quantity, unit_price, total}], "
        "subtotal: string, tax: string, total: string}. "
        "Use null for any field not found. Return only the JSON, no explanation."
    ),
    "chart": (
        "Analyse this chart and extract: "
        "1. Chart type (bar, line, pie, etc) "
        "2. Title "
        "3. X-axis label and values "
        "4. Y-axis label and range "
        "5. Data series with their values "
        "6. Key trend or insight. "
        "Format clearly with headers."
    ),
    "form": (
        "Extract all form fields and their values from this image. "
        "Return as valid JSON: {field_name: value, ...}. "
        "For empty fields use null. Return only the JSON."
    ),
    "table": (
        "Extract all tabular data from this image as CSV. "
        "First row should be the header. Use comma as delimiter. "
        "Return only the CSV content, no explanation."
    ),
}


def load_image_b64(path: str) -> tuple[str, int]:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8"), len(data)


def extract(model: str, image_b64: str, mode: str) -> str:
    prompt = PROMPTS[mode]
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def display_result(text: str, mode: str):
    if mode in ("invoice", "form"):
        # try to pretty-print as JSON
        try:
            # strip any markdown code fences the model may add
            clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            console.print(Syntax(json.dumps(parsed, indent=2), "json", theme="monokai"))
            return
        except json.JSONDecodeError:
            pass  # fall through to plain text
    elif mode == "table":
        console.print(Syntax(text.strip(), "text", theme="monokai"))
        return

    console.print(text.strip())


def main():
    parser = argparse.ArgumentParser(description="OCR and structured data extraction from images")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--mode",
        choices=list(PROMPTS.keys()),
        default="text",
        help="Extraction mode (default: text)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Vision model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    console.print(f"\n[bold]Image :[/bold] {args.image}")
    console.print(f"[bold]Mode  :[/bold] {args.mode}")
    console.print(f"[bold]Model :[/bold] {args.model}\n")
    console.print("─" * 60)

    image_b64, size_bytes = load_image_b64(args.image)
    console.print(f"[dim]Image: {size_bytes / 1024:.0f} KB → {len(image_b64) // 1024} KB base64[/dim]\n")

    result = extract(args.model, image_b64, args.mode)
    display_result(result, args.mode)


if __name__ == "__main__":
    main()
