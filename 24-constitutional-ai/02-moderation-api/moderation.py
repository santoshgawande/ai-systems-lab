"""
OpenAI Moderation API: classify text into harm categories with confidence scores.
Use as a fast pre-filter before your main LLM call.
"""
import os
import json

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(OPENAI_KEY)
except ImportError:
    OPENAI_AVAILABLE = False

# ─── Categories returned by the moderation API ───────────────────────────────

CATEGORIES = {
    "harassment":              "Harassing, bullying, or threatening content",
    "harassment/threatening":  "Harassment with threats of violence",
    "hate":                    "Content promoting hatred based on protected characteristics",
    "hate/threatening":        "Hate content with threats",
    "illicit":                 "Content facilitating illegal activities",
    "illicit/violent":         "Violent illegal activities",
    "self-harm":               "Content encouraging self-harm or suicide",
    "self-harm/intent":        "Expressed intent to self-harm",
    "self-harm/instructions":  "Instructions for self-harm",
    "sexual":                  "Sexually explicit content",
    "sexual/minors":           "Sexual content involving minors",
    "violence":                "Violent content",
    "violence/graphic":        "Graphic or gratuitous violence",
}

# ─── Test inputs ──────────────────────────────────────────────────────────────

TEST_INPUTS = [
    {"label": "Safe — recipe",      "text": "How do I make chocolate chip cookies?"},
    {"label": "Safe — code",        "text": "Write a Python function to sort a list."},
    {"label": "Safe — medical",     "text": "What are common symptoms of a cold?"},
    {"label": "Borderline — anger", "text": "I'm so angry I could scream. My coworker keeps taking credit for my work."},
    {"label": "Flagged — threat",   "text": "I want to hurt the person who stole my bike."},
    {"label": "Flagged — hate",     "text": "People from [group] are all criminals and should be deported."},
]

# ─── Moderation ───────────────────────────────────────────────────────────────

def moderate(text: str, client) -> dict:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    r = result.results[0]
    scores = {k: round(v, 4) for k, v in r.category_scores.__dict__.items()}
    flags = {k: v for k, v in r.categories.__dict__.items() if v}
    return {
        "flagged": r.flagged,
        "flags": flags,
        "scores": scores,
        "top_category": max(scores, key=scores.get),
        "top_score": max(scores.values()),
    }


def pipeline_demo(client) -> None:
    """Show how moderation fits into an LLM pipeline."""
    import httpx

    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    def safe_complete(user_input: str) -> str:
        # Step 1: moderate input
        result = moderate(user_input, client)
        if result["flagged"]:
            top = result["top_category"]
            return f"[BLOCKED] Content policy violation: {top} (score={result['top_score']:.3f})"

        # Step 2: call LLM (only if clean)
        if ANTHROPIC_KEY:
            try:
                r = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 80,
                        "messages": [{"role": "user", "content": user_input}],
                    },
                    timeout=15,
                )
                return r.json()["content"][0]["text"].strip()
            except Exception:
                pass
        return "[LLM not configured — set ANTHROPIC_API_KEY]"

    inputs = [
        "What is the capital of France?",
        "Write a Python hello world.",
        "I want to attack someone at the grocery store.",
    ]

    print("─── Moderation-gated pipeline ───\n")
    for inp in inputs:
        print(f"Input:  {inp}")
        result = safe_complete(inp)
        print(f"Output: {result}")
        print()


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== OPENAI MODERATION API DEMO ===\n")

if not OPENAI_AVAILABLE:
    if not OPENAI_KEY:
        reason = "OPENAI_API_KEY not set"
    else:
        reason = "pip install openai"
    print(f"Skipping live demo ({reason})\n")
    print("""
Key API:

from openai import OpenAI
client = OpenAI()

result = client.moderations.create(
    model="omni-moderation-latest",  # or "text-moderation-latest"
    input="I want to harm someone.",
)
r = result.results[0]

print(r.flagged)                     # True/False
print(r.categories.violence)         # True if violence detected
print(r.category_scores.violence)    # 0.0 - 1.0 confidence

# Typical pipeline:
def safe_complete(user_input):
    moderation = client.moderations.create(input=user_input)
    if moderation.results[0].flagged:
        return "I can't help with that."
    return llm_complete(user_input)

Categories (omni-moderation-latest):
  harassment, harassment/threatening
  hate, hate/threatening
  illicit, illicit/violent
  self-harm, self-harm/intent, self-harm/instructions
  sexual, sexual/minors
  violence, violence/graphic

Cost: $0 — moderation endpoint is FREE with any OpenAI API key.
Latency: ~100-200ms (fast, run synchronously before main LLM call)
""")
    raise SystemExit(0)

client = OpenAI(api_key=OPENAI_KEY)
print(f"Using model: omni-moderation-latest\n")
print(f"{'Input':<40} {'Flagged':<8} {'Top Category':<30} {'Score'}")
print("─" * 90)

for item in TEST_INPUTS:
    result = moderate(item["text"], client)
    flagged_str = "YES ⚠" if result["flagged"] else "no"
    print(f"{item['label']:<40} {flagged_str:<8} {result['top_category']:<30} {result['top_score']:.4f}")

print()

# Show full scores for a flagged example
flagged_items = [i for i in TEST_INPUTS if "Flagged" in i["label"]]
if flagged_items:
    item = flagged_items[0]
    print(f"Detailed scores for: {item['text'][:60]!r}")
    result = moderate(item["text"], client)
    significant = {k: v for k, v in result["scores"].items() if v > 0.001}
    for cat, score in sorted(significant.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"  {cat:<35} {score:.4f}  {bar}")

print()
pipeline_demo(client)
