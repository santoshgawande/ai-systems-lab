"""
Zero-shot vs few-shot prompting.
Shows how examples in context enforce output format and improve consistency.
"""
import httpx

OLLAMA = "http://localhost:11434"
MODEL = "phi4"  # small model shows the effect more clearly

INPUTS = [
    "I absolutely love this product! Works perfectly.",
    "Complete waste of money. Broke on day one.",
    "It's okay. Does what it says, nothing special.",
    "Incredible quality — I recommend it to everyone.",
    "Never buying from this brand again.",
]


def ask(system: str, user: str) -> str:
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ─── Strategy 1: Zero-shot ────────────────────────────────────────────────────
ZERO_SHOT = "Classify the sentiment of this text."

print("=== ZERO-SHOT ===")
print(f"System: {ZERO_SHOT!r}\n")
for text in INPUTS:
    result = ask(ZERO_SHOT, text)
    print(f"  Input:  {text[:50]!r}")
    print(f"  Output: {result[:80]!r}\n")


# ─── Strategy 2: Few-shot ─────────────────────────────────────────────────────
FEW_SHOT = """Classify the sentiment of the text.
Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.

Examples:
"Exceeded all my expectations, really happy!" → POSITIVE
"Terrible product, broke immediately." → NEGATIVE
"It arrived on time. Average quality." → NEUTRAL
"The best purchase I've made this year!" → POSITIVE
"Stopped working after 3 days." → NEGATIVE"""

print("\n=== FEW-SHOT (5 examples) ===")
print("System includes 5 labeled examples\n")
for text in INPUTS:
    result = ask(FEW_SHOT, text)
    parseable = result.strip().upper() in ("POSITIVE", "NEGATIVE", "NEUTRAL")
    icon = "✓" if parseable else "✗"
    print(f"  {icon} {result:<10}  ← {text[:50]!r}")

print()
print("Key insight: few-shot output is directly parseable.")
print("Zero-shot output varies — 'This is positive!', 'Sentiment: positive', etc.")


# ─── Strategy 3: Few-shot extraction ─────────────────────────────────────────
print("\n\n=== FEW-SHOT EXTRACTION (structured output) ===")

EXTRACT_SYSTEM = """Extract product, price, and sentiment from the review.
Respond with: PRODUCT | PRICE | SENTIMENT
SENTIMENT must be: POSITIVE, NEGATIVE, or NEUTRAL. Use N/A if not mentioned.

Examples:
"The AirPods at $249 are worth every penny!" → AirPods | $249 | POSITIVE
"Broken USB cable I bought for $12." → USB cable | $12 | NEGATIVE
"Good laptop. No price given in review." → laptop | N/A | POSITIVE"""

REVIEWS = [
    "The Sony WH-1000XM5 headphones at $350 are absolutely incredible.",
    "Bought this $25 phone case. It cracked after a week.",
    "Nice keyboard. Didn't see the price but seems reasonable.",
]

for review in REVIEWS:
    result = ask(EXTRACT_SYSTEM, review)
    parts = result.split("|")
    parseable = len(parts) == 3
    icon = "✓" if parseable else "✗"
    print(f"  {icon} {result}")
