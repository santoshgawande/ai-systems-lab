#!/usr/bin/env python3
"""
mini-copilot: terminal code completion using fill-in-the-middle (FIM) prompting.
Modes: complete (FIM), implement (function body), test (generate tests).
Uses Ollama locally — no API key needed.
"""
import os
import sys
import httpx
import argparse

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
MODEL = os.environ.get("COPILOT_MODEL", "codellama")  # or deepseek-coder, qwen2.5-coder
FALLBACK_MODEL = "llama3.2"


def ollama_generate(prompt: str, system: str = "", model: str = MODEL) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    try:
        r = httpx.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        # Try fallback model
        if model != FALLBACK_MODEL:
            return ollama_generate(prompt, system, FALLBACK_MODEL)
        return f"Error: {e}"


def stream_generate(prompt: str, system: str = "", model: str = MODEL):
    payload = {"model": model, "prompt": prompt, "system": system, "stream": True}
    try:
        with httpx.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60) as r:
            for line in r.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except Exception as e:
        yield f"\nError: {e}"


# ─── FIM (fill-in-middle) completion ─────────────────────────────────────────

FIM_SYSTEM = """You are a code completion assistant.
Given the code BEFORE and AFTER the cursor, output ONLY the code that goes between them.
No explanations. No markdown. No comments about what you're doing.
Just the exact code that completes the gap."""

def fim_complete(before: str, after: str) -> str:
    prompt = f"PREFIX:\n{before}\n\nSUFFIX:\n{after}\n\nCOMPLETION (only the code, nothing else):"
    return ollama_generate(prompt, FIM_SYSTEM)


# ─── Function implementation ──────────────────────────────────────────────────

IMPL_SYSTEM = """You are a coding assistant. Implement the function body.
Output ONLY the function body (no function signature).
Use the correct language. Be concise. No markdown code blocks."""

def implement_function(signature: str, docstring: str = "", language: str = "python") -> str:
    lang_hint = f"Language: {language}" if language else ""
    prompt = f"""{lang_hint}
Function signature:
{signature}

{f"Docstring: {docstring}" if docstring else ""}

Implement the function body only (not the signature):"""
    return ollama_generate(prompt, IMPL_SYSTEM)


# ─── Test generation ─────────────────────────────────────────────────────────

TEST_SYSTEM = """You are a testing expert. Generate comprehensive unit tests.
Output ONLY the test code. No explanations. Use pytest for Python."""

def generate_tests(code: str, language: str = "python") -> str:
    prompt = f"""Generate unit tests for this {language} code:

{code}

Tests (pytest style, with edge cases):"""
    return ollama_generate(prompt, TEST_SYSTEM)


# ─── Docstring generation ────────────────────────────────────────────────────

DOC_SYSTEM = "Generate a concise docstring. One line summary, then Args/Returns if complex. No fluff."

def generate_docstring(function_code: str, language: str = "python") -> str:
    prompt = f"Generate a docstring for this {language} function:\n\n{function_code}\n\nDocstring:"
    return ollama_generate(prompt, DOC_SYSTEM)


# ─── Interactive REPL ────────────────────────────────────────────────────────

HELP_TEXT = """
mini-copilot commands:
  fim     — fill in the middle (enter prefix, then suffix)
  impl    — implement a function from signature
  test    — generate tests for a function
  doc     — generate docstring for a function
  help    — show this help
  quit    — exit

Or paste code directly and press Enter twice for completion.
"""

def read_multiline(prompt_text: str = "") -> str:
    if prompt_text:
        print(prompt_text)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            if lines and lines[-1] == "":
                lines.pop()
                break
            lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


def run_interactive():
    print("=== MINI-COPILOT ===")
    print(f"Model: {MODEL} (fallback: {FALLBACK_MODEL})")
    print("Type 'help' for commands. Ctrl+C to exit.\n")

    while True:
        try:
            cmd = input("copilot> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not cmd:
            continue

        elif cmd in ("quit", "exit", "q"):
            print("Bye!")
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "fim":
            print("Enter code BEFORE cursor (blank line to finish):")
            before = read_multiline()
            print("Enter code AFTER cursor (blank line to finish):")
            after = read_multiline()
            print("\nCompletion:")
            print("─" * 50)
            for token in stream_generate(
                f"PREFIX:\n{before}\n\nSUFFIX:\n{after}\n\nCOMPLETION:",
                system=FIM_SYSTEM
            ):
                print(token, end="", flush=True)
            print("\n" + "─" * 50)

        elif cmd == "impl":
            print("Enter function signature (e.g. 'def search(query: str) -> list[str]:'):")
            sig = input().strip()
            print("Enter docstring (optional, Enter to skip):")
            doc = input().strip()
            print("\nImplementation:")
            print("─" * 50)
            for token in stream_generate(
                f"Function signature:\n{sig}\n{f'Docstring: {doc}' if doc else ''}\n\nImplementation:",
                system=IMPL_SYSTEM
            ):
                print(token, end="", flush=True)
            print("\n" + "─" * 50)

        elif cmd == "test":
            print("Paste function code (blank line twice to finish):")
            code = read_multiline()
            print("\nTests:")
            print("─" * 50)
            for token in stream_generate(
                f"Generate pytest tests for:\n\n{code}\n\nTests:",
                system=TEST_SYSTEM
            ):
                print(token, end="", flush=True)
            print("\n" + "─" * 50)

        elif cmd == "doc":
            print("Paste function code (blank line twice to finish):")
            code = read_multiline()
            print("\nDocstring:")
            print("─" * 50)
            for token in stream_generate(
                f"Generate a concise docstring for:\n\n{code}\n\nDocstring:",
                system=DOC_SYSTEM
            ):
                print(token, end="", flush=True)
            print("\n" + "─" * 50)

        else:
            # Treat as code prefix for direct completion
            print("Completing...")
            print("─" * 50)
            for token in stream_generate(
                f"Continue this code:\n\n{cmd}\n",
                system="Complete the code. Output only the completion, nothing else."
            ):
                print(token, end="", flush=True)
            print("\n" + "─" * 50)


# ─── Batch demo (non-interactive) ────────────────────────────────────────────

def run_demo():
    print("=== MINI-COPILOT DEMO ===\n")

    # FIM demo
    print("─── FIM (fill-in-middle) ───\n")
    before = """def calculate_stats(numbers: list[float]) -> dict:
    if not numbers:
        return {}
    total = sum(numbers)
    mean = total / len(numbers)"""
    after = """
    return {
        "count": len(numbers),
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "total": total,
    }"""
    print(f"Before:\n{before}\n\nAfter:\n{after}\n")
    completion = fim_complete(before, after)
    print(f"Completion:\n{completion}\n")

    # Implementation demo
    print("─── Function implementation ───\n")
    sig = "def retry(func, max_retries: int = 3, delay: float = 1.0):"
    doc = "Retry func up to max_retries times with exponential backoff. Returns result or raises last exception."
    print(f"Signature: {sig}")
    print(f"Docstring: {doc}\n")
    impl = implement_function(sig, doc)
    print(f"Implementation:\n{impl}\n")

    # Test generation demo
    print("─── Test generation ───\n")
    func_code = """def is_valid_email(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))"""
    print(f"Function:\n{func_code}\n")
    tests = generate_tests(func_code)
    print(f"Tests:\n{tests}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="mini-copilot: terminal code completion")
parser.add_argument("--demo", action="store_true", help="Run batch demo instead of interactive mode")
args = parser.parse_args()

# Check Ollama
try:
    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
except Exception:
    print(f"Ollama not running at {OLLAMA_BASE}")
    print("Start Ollama: ollama serve")
    print("Pull a code model: ollama pull codellama  or  ollama pull qwen2.5-coder:7b")
    sys.exit(1)

if args.demo:
    run_demo()
else:
    run_interactive()
