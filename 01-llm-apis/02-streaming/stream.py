"""
Lab 02 — Streaming

Shows how LLMs send tokens one at a time over HTTP (same as ChatGPT/Claude).
Each line from the server is a JSON chunk with a partial token.

Run: python stream.py
Run with custom prompt: python stream.py "What is RAG?"
"""

import json
import sys
import time
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.3:70b"


def stream_chat(prompt: str, model: str = MODEL):
    """Yields (token, is_done, stats) tuples as they arrive from the server."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    with requests.post(
        f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk = json.loads(raw_line)
            token = chunk.get("message", {}).get("content", "")
            done = chunk.get("done", False)
            yield token, done, chunk


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"

    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")
    print("-" * 60)

    start = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    for token, done, chunk in stream_chat(prompt):
        if token:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(token, end="", flush=True)
            total_tokens += 1

        if done:
            elapsed = time.perf_counter() - start
            ttft = (first_token_time - start) * 1000 if first_token_time else 0
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

            print(f"\n\n{'─' * 60}")
            print(f"Time to first token : {ttft:.0f} ms")
            print(f"Total latency       : {elapsed * 1000:.0f} ms")
            print(f"Output tokens       : {total_tokens}")
            print(f"Throughput          : {tokens_per_sec:.1f} tokens/sec")
            break


if __name__ == "__main__":
    main()
