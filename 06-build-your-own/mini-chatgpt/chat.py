#!/usr/bin/env python3
"""mini-chatgpt: streaming multi-turn chat with persistent memory and context management."""

import json
import time
import httpx
from pathlib import Path
from datetime import datetime

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"
MAX_CONTEXT_TOKENS = 3000
HISTORY_FILE = Path("/tmp/mini_chatgpt_history.json")

SYSTEM = "You are a helpful, concise AI assistant. Be direct and clear."


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def trim_to_budget(history: list[dict], budget: int = MAX_CONTEXT_TOKENS) -> tuple[list[dict], int]:
    """Keep the most recent messages that fit within the token budget."""
    kept, total = [], 0
    for msg in reversed(history):
        t = token_estimate(msg["content"])
        if total + t > budget:
            break
        kept.insert(0, msg)
        total += t
    return kept, len(history) - len(kept)


def stream_response(messages: list[dict]) -> tuple[str, dict]:
    parts: list[str] = []
    usage: dict = {}
    with httpx.stream("POST", f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": messages,
        "stream": True,
    }, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                parts.append(token)
            if chunk.get("done"):
                usage = {
                    "in": chunk.get("prompt_eval_count", 0),
                    "out": chunk.get("eval_count", 0),
                }
    print()
    return "".join(parts), usage


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def save_history(history: list[dict]):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def main():
    history = load_history()
    print(f"mini-chatgpt  model={MODEL}")
    print(f"History: {len(history)} messages  (persisted at {HISTORY_FILE})")
    print("Commands: /clear  /history  quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input == "/clear":
            history = []
            save_history(history)
            print("History cleared.\n")
            continue

        if user_input == "/history":
            if not history:
                print("(empty)\n")
            for msg in history[-10:]:
                role = "You" if msg["role"] == "user" else "Assistant"
                print(f"  {role}: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
            print()
            continue

        history.append({"role": "user", "content": user_input})

        trimmed, dropped = trim_to_budget(history[:-1])
        if dropped > 0:
            print(f"  [context: {dropped} older messages trimmed to fit {MAX_CONTEXT_TOKENS} token budget]\n")

        messages = [{"role": "system", "content": SYSTEM}] + trimmed + [history[-1]]

        print("Assistant: ", end="", flush=True)
        start = time.time()

        try:
            response, usage = stream_response(messages)
        except httpx.TimeoutException:
            print("\n[timeout]\n")
            history.pop()
            continue
        except Exception as e:
            print(f"\n[error: {e}]\n")
            history.pop()
            continue

        elapsed = time.time() - start
        print(f"  [{elapsed:.1f}s | {usage.get('in', 0)} in / {usage.get('out', 0)} out tokens]\n")

        history.append({"role": "assistant", "content": response})
        save_history(history)


if __name__ == "__main__":
    main()
