"""
Lab 01 — Tokenizer basics

Answers: what IS a token, why does the same sentence have different counts
across models, and why token count != word count.
"""

import tiktoken
from rich.console import Console
from rich.table import Table

console = Console()


def show_token_splits(text: str, enc: tiktoken.Encoding) -> None:
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    console.print(f"\n[bold]Text:[/bold] {text!r}")
    console.print(f"[bold]Token count:[/bold] {len(tokens)}")
    console.print(f"[bold]Splits:[/bold] {decoded}")


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    console.rule("[bold blue]1. Token splits — same sentence, different lenses")

    samples = [
        "Hello, world!",
        "What is the weather in San Francisco?",
        "Calculate 123 * 456 + 789",
        # JSON (like a tool definition) is expensive
        '{"type": "function", "name": "get_weather", "parameters": {}}',
        # Code tokenizes differently than prose
        "def get_weather(city: str) -> dict:",
        # Whitespace matters
        "    " * 10 + "deeply nested code",
    ]

    for s in samples:
        show_token_splits(s, enc)

    console.rule("[bold blue]2. Why JSON tool schemas are expensive")

    simple_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }

    import json
    tool_json = json.dumps(simple_tool, indent=2)
    tokens = enc.encode(tool_json)
    console.print(f"\nA single simple tool definition = [bold red]{len(tokens)} tokens[/bold red]")
    console.print(f"That's the same as ~{len(tokens)//5} words of plain English.\n")
    console.print("[dim]Multiply by 5-10 tools and this is your hidden overhead.[/dim]")

    console.rule("[bold blue]3. Token count across different text types")

    table = Table(show_header=True)
    table.add_column("Text type")
    table.add_column("Example (truncated)", max_width=40)
    table.add_column("Tokens", justify="right")
    table.add_column("~Words", justify="right")
    table.add_column("Tokens/Word", justify="right")

    cases = [
        ("Plain English", "The quick brown fox jumps over the lazy dog and then runs away"),
        ("Python code", "def calculate(x: int, y: int) -> int:\n    return x * y + x"),
        ("JSON schema", json.dumps(simple_tool)),
        ("Repeated whitespace", "    " * 20),
        ("Numbers", "123456789 987654321 111222333 444555666"),
        ("URL", "https://api.example.com/v2/weather?city=sf&units=metric&format=json"),
    ]

    for label, text in cases:
        toks = len(enc.encode(text))
        words = len(text.split())
        ratio = f"{toks/max(words,1):.1f}x" if words else "N/A"
        table.add_row(label, text[:40] + ("…" if len(text) > 40 else ""), str(toks), str(words), ratio)

    console.print(table)
    console.print("\n[yellow]Key insight:[/yellow] JSON (tool schemas) tokenizes at ~1.5-2x the rate of plain English.\n"
                  "URLs and code have irregular splits. This is why MCP overhead is non-trivial.")


if __name__ == "__main__":
    main()
