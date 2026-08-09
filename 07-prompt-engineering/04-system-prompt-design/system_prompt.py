"""
System prompt design — how structure and precision affect output consistency.
Compares weak / medium / strong prompts on the same task.
"""
import httpx

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


TASK = "How should I handle database connection failures in a Spring Boot app?"

# ─── Three quality levels ─────────────────────────────────────────────────────

WEAK = "You are a helpful assistant."

MEDIUM = "You are a senior software engineer. Answer technical questions clearly and concisely."

STRONG = """You are a senior backend engineer with 12 years of production Java experience.

Format rules (follow exactly):
1. Start with the recommended approach in 1-2 sentences
2. Show a code example (Spring Boot Java)
3. End with one "Watch out:" line about the most common mistake

Length: 150-250 words total.
Tone: direct, as if reviewing a colleague's PR.
Never add preamble like "Great question!" or "Of course!"."""

print("=== SYSTEM PROMPT COMPARISON ===")
print(f"Task: {TASK!r}\n")

for label, system in [("WEAK", WEAK), ("MEDIUM", MEDIUM), ("STRONG", STRONG)]:
    result = ask(system, TASK)
    print(f"─── {label} ({len(result)} chars) ───")
    print(f"System: {system[:80]!r}")
    print()
    print(result[:500])
    print()


# ─── Constraint enforcement: length ──────────────────────────────────────────
print("\n=== CONSTRAINT: ENFORCE LENGTH ===\n")

STRICT_SHORT = "Answer in exactly one sentence. No more."
STRICT_BULLETS = "Answer in exactly 3 bullet points. Each bullet is one sentence. No intro, no outro."

questions = [
    "What is Redis?",
    "Why use microservices?",
]

for q in questions:
    short = ask(STRICT_SHORT, q)
    bullets = ask(STRICT_BULLETS, q)
    print(f"Q: {q}")
    print(f"  One sentence:  {short[:100]!r}")
    bullet_count = short.count("\n") + 1
    print(f"  3 bullets:     {bullets[:200]!r}")
    print()


# ─── Persona consistency ──────────────────────────────────────────────────────
print("=== PERSONA CONSISTENCY ===\n")

SKEPTIC = """You are a pragmatic senior engineer who is skeptical of hype.
When asked about technology choices, always mention: the failure mode, the operational complexity, and whether a simpler solution would work.
Be direct. No enthusiasm. No marketing language."""

EVANGELIST = """You are an enthusiastic technology advocate.
When asked about technology choices, highlight the benefits, the community, and exciting use cases.
Be positive and energetic."""

opinion_q = "Should I use Kubernetes for my startup?"

skeptic_ans = ask(SKEPTIC, opinion_q)
evangelist_ans = ask(EVANGELIST, opinion_q)

print(f"Q: {opinion_q}\n")
print(f"Skeptic:    {skeptic_ans[:300]!r}\n")
print(f"Evangelist: {evangelist_ans[:300]!r}\n")
print("Same question, opposite personas → completely different guidance.")
print("Your system prompt IS the product. Design it with the same care as code.")
