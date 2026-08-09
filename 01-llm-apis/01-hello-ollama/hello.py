"""
Lab 01 — Hello Ollama

Raw HTTP call to a local model. Inspect the full request and response.
No SDK, no framework — just requests.

Run: python hello.py
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.3:70b"  # change to any model you have: `ollama list`


def chat(prompt: str, model: str = MODEL) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    prompt = "Explain what a transformer model is in 3 sentences."

    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")
    print("-" * 60)

    result = chat(prompt)

    # what the response looks like
    print("Full response JSON:")
    print(json.dumps(result, indent=2))
    print("-" * 60)

    # the parts you actually care about
    message = result["message"]["content"]
    tokens_prompt = result.get("prompt_eval_count", 0)
    tokens_output = result.get("eval_count", 0)
    duration_ms = result.get("total_duration", 0) / 1_000_000

    print(f"\nAnswer:\n{message}")
    print(f"\nTokens — prompt: {tokens_prompt}, output: {tokens_output}")
    print(f"Latency: {duration_ms:.0f} ms")


if __name__ == "__main__":
    main()
