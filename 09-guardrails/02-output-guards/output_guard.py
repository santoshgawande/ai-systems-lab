"""
Output guards: validate and filter LLM responses before they reach your application.
Covers JSON validation, hallucination detection, and unsafe content filtering.
"""
import re
import json
import httpx
from dataclasses import dataclass, field
from typing import Optional

OLLAMA = "http://localhost:11434"
MODEL = "phi4"

# ─── Hallucination signal patterns ───────────────────────────────────────────
HALLUCINATION_SIGNALS = [
    re.compile(r"as of my (knowledge|training) (cutoff|date)", re.I),
    re.compile(r"I (don't|do not|cannot) have (access|information) (to|about)", re.I),
    re.compile(r"I('m| am) not sure (but|if)", re.I),
    re.compile(r"you (can|could|should) (find|check|verify) (this|that) (at|on|in)", re.I),
    re.compile(r"https?://[^\s]+", re.I),          # URLs in output (often hallucinated)
    re.compile(r"\b(approximately|roughly|around)\s+\d", re.I),  # fuzzy numbers
]

UNSAFE_PATTERNS = [
    re.compile(r"\b(how to (make|build|create|synthesize))\b.{0,30}(bomb|weapon|explosive|poison)", re.I),
    re.compile(r"\b(step \d+|first|next|then|finally).{0,50}(illegal|dangerous|harmful)", re.I),
]


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hallucination_signals: list[str] = field(default_factory=list)


def validate_json(text: str, required_keys: list[str]) -> ValidationResult:
    """Check that output is valid JSON with all required keys."""
    errors, warnings = [], []

    parsed = None
    for extractor in [
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find("{"):t.rfind("}") + 1]) if "{" in t else (_ for _ in ()).throw(ValueError()),
    ]:
        try:
            parsed = extractor(text)
            break
        except Exception:
            pass

    if parsed is None:
        errors.append("Output is not valid JSON")
        return ValidationResult(ok=False, errors=errors)

    missing = [k for k in required_keys if k not in parsed]
    if missing:
        errors.append(f"Missing required keys: {missing}")

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def check_hallucination(text: str) -> list[str]:
    """Flag phrases that often accompany hallucinated content."""
    found = []
    for pattern in HALLUCINATION_SIGNALS:
        match = pattern.search(text)
        if match:
            found.append(match.group(0))
    return found


def check_unsafe(text: str) -> bool:
    """Quick scan for unsafe content patterns."""
    return any(p.search(text) for p in UNSAFE_PATTERNS)


def validate_output(text: str, required_json_keys: Optional[list[str]] = None) -> ValidationResult:
    result = ValidationResult(ok=True)

    # JSON validation
    if required_json_keys:
        json_result = validate_json(text, required_json_keys)
        result.errors.extend(json_result.errors)
        if json_result.errors:
            result.ok = False

    # Hallucination check
    signals = check_hallucination(text)
    if signals:
        result.hallucination_signals = signals
        result.warnings.append(f"Hallucination signals: {signals[:2]}")

    # Safety check
    if check_unsafe(text):
        result.ok = False
        result.errors.append("Unsafe content detected")

    # Length sanity check
    if len(text.strip()) < 5:
        result.ok = False
        result.errors.append("Response too short (possible refusal)")

    return result


def ask_with_guard(system: str, user: str, required_json_keys: Optional[list[str]] = None, max_retries: int = 2) -> tuple[str, ValidationResult]:
    """Call LLM with automatic retry on output validation failure."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(1, max_retries + 2):
        r = httpx.post(f"{OLLAMA}/api/chat",
                       json={"model": MODEL, "messages": messages, "stream": False},
                       timeout=60)
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
        result = validate_output(reply, required_json_keys)

        if result.ok:
            return reply, result

        if attempt <= max_retries:
            error_feedback = "; ".join(result.errors)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Error in your response: {error_feedback}. Please fix and respond again."})

    return reply, result


# ─── Demo ─────────────────────────────────────────────────────────────────────
JSON_SYSTEM = 'Extract product, price, and rating from the text. Respond ONLY with JSON: {"product": "...", "price": 0.0, "rating": 1-5}'
FACT_SYSTEM = "You are a factual assistant. Answer concisely."
UNSAFE_SYSTEM = "You are a helpful assistant."

print("=== OUTPUT VALIDATION DEMO ===\n")

tests = [
    ("JSON validation — valid",   JSON_SYSTEM, "The laptop costs $999 and I give it 4 stars.", ["product", "price", "rating"]),
    ("JSON validation — retry",   JSON_SYSTEM, "Extract info from: great keyboard, paid $50, loved it.", ["product", "price", "rating"]),
    ("Hallucination detection",   FACT_SYSTEM, "What is the current stock price of Apple?", None),
    ("Fact with uncertainty",     FACT_SYSTEM, "What is the exact population of Tokyo right now?", None),
]

for label, system, user, keys in tests:
    output, result = ask_with_guard(system, user, required_json_keys=keys)
    status = "✓" if result.ok else "✗"
    print(f"{status} {label}")
    print(f"   Output:  {output[:100]!r}")
    if result.errors:
        print(f"   Errors:  {result.errors}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    if result.hallucination_signals:
        print(f"   Hallucination signals: {result.hallucination_signals}")
    print()
