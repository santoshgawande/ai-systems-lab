"""
Getting reliable structured (JSON) output from LLMs.
Three strategies: vague, schema-specified, and with retry on parse failure.
"""
import json
import httpx
from typing import Optional

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"


def ask(system: str, user: str) -> str:
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def extract_json(text: str) -> Optional[dict]:
    """Parse JSON from model output that may include markdown or prose."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fence
    for marker in ["```json\n", "```\n", "```json", "```"]:
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end >= 0:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def ask_json(system: str, user: str, required_keys: list[str], max_retries: int = 2) -> Optional[dict]:
    """Ask with automatic retry if JSON parse fails."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(1, max_retries + 2):
        r = httpx.post(f"{OLLAMA}/api/chat", json={
            "model": MODEL, "messages": messages, "stream": False,
        }, timeout=60)
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
        parsed = extract_json(reply)

        if parsed and all(k in parsed for k in required_keys):
            return parsed

        if attempt <= max_retries:
            # Feed parse error back as context for retry
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Your response could not be parsed as JSON with keys {required_keys}. Respond with valid JSON only."})
    return None


REVIEWS = [
    "The MacBook Pro M4 is incredibly fast. Battery lasts 20 hours. $2,499 — steep but worth it for professionals.",
    "Terrible headphones. Sound quality is muddy and the right ear died after 2 weeks. Would NOT recommend.",
    "Decent laptop for the price. 8GB RAM feels limiting for dev work. Beautiful display though. Probably a 3 out of 5.",
]

SCHEMA = """{
  "product": "product name (string)",
  "sentiment": "POSITIVE | NEGATIVE | NEUTRAL",
  "rating": integer 1-5 (infer from context if not explicit),
  "pros": ["array of positive aspects"],
  "cons": ["array of negative aspects"],
  "price_mentioned": true | false
}"""

if __name__ == "__main__":
    # ─── Strategy 1: Vague instruction ────────────────────────────────────────────
    VAGUE = "Extract product info from the review as JSON."

    print("=== STRATEGY 1: Vague instruction ===\n")
    for review in REVIEWS[:2]:
        result = ask(VAGUE, review)
        parsed = extract_json(result)
        icon = "✓" if parsed else "✗"
        print(f"  {icon} Raw: {result[:100]!r}")
        print()

    # ─── Strategy 2: Precise schema ───────────────────────────────────────────────
    PRECISE = f"""Extract product information from the review.
Respond ONLY with valid JSON matching this schema — no other text, no markdown:

{SCHEMA}"""

    print("=== STRATEGY 2: Precise schema ===\n")
    for review in REVIEWS:
        result = ask(PRECISE, review)
        parsed = extract_json(result)
        icon = "✓" if parsed else "✗"
        if parsed:
            print(f"  {icon} sentiment={parsed.get('sentiment')!r}  rating={parsed.get('rating')}  price={parsed.get('price_mentioned')}")
        else:
            print(f"  {icon} Parse failed: {result[:80]!r}")

    # ─── Strategy 3: Schema + retry on parse failure ──────────────────────────────
    print("\n=== STRATEGY 3: Schema + auto-retry ===\n")
    for review in REVIEWS:
        parsed = ask_json(PRECISE, review, required_keys=["product", "sentiment", "rating"])
        if parsed:
            print(f"  ✓ {parsed.get('product')!r}  {parsed.get('sentiment')}  {parsed.get('rating')}/5")
        else:
            print(f"  ✗ Failed after retries")
