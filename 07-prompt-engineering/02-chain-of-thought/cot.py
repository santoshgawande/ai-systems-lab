"""
Chain-of-thought prompting.
Demonstrates how "think step by step" improves accuracy on reasoning tasks.
"""
import httpx

OLLAMA = "http://localhost:11434"
MODEL = "phi4"


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


PROBLEMS = [
    {
        "question": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "correct": "5 cents / $0.05",
        "trap": "Most people instinctively say 10 cents — but that makes the bat $1.10, total $1.20.",
    },
    {
        "question": "A doctor gives you 3 pills and tells you to take one every 30 minutes. How many minutes until all pills are taken?",
        "correct": "60 minutes",
        "trap": "3 pills × 30 min = 90 min is wrong. You take pill 1 at 0 min, pill 2 at 30 min, pill 3 at 60 min.",
    },
    {
        "question": "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons of water?",
        "correct": "Fill 5-gal, pour into 3-gal (leaves 2), empty 3-gal, pour 2 into 3-gal, fill 5-gal again, pour into 3-gal (3-gal needs 1 more) → 4 gallons remain in 5-gal jug",
        "trap": "Requires multi-step reasoning — direct answer rarely works.",
    },
]

DIRECT = "Answer concisely. Give only the final answer."
ZERO_COT = "Think through this step by step, then give your final answer."
FEW_SHOT_COT = """Solve math and logic problems step by step.

Example:
Q: A train travels 60 mph for 2.5 hours. How far does it go?
A: Step 1: distance = speed × time
   Step 2: distance = 60 × 2.5 = 150
   Final answer: 150 miles"""

for p in PROBLEMS:
    print(f"Problem: {p['question']}")
    print(f"Correct: {p['correct']}")
    print(f"Trap:    {p['trap']}\n")

    direct_ans = ask(DIRECT, p["question"])
    cot_ans = ask(ZERO_COT, p["question"])

    print(f"Direct:         {direct_ans[:120]!r}")
    print(f"With CoT:       {cot_ans[:250]!r}")
    print("=" * 70)
    print()

# ─── When CoT hurts (adds noise to simple tasks) ─────────────────────────────
print("\n=== CoT HURTS on simple tasks ===")
print("For factual recall, CoT just adds tokens without improving accuracy.\n")

simple = [
    "What is the capital of Japan?",
    "What does HTML stand for?",
    "What year did World War II end?",
]

for q in simple:
    direct = ask(DIRECT, q)
    cot = ask(ZERO_COT, q)
    print(f"Q: {q}")
    print(f"  Direct ({len(direct)} chars): {direct[:60]!r}")
    print(f"  CoT    ({len(cot)} chars): {cot[:60]!r}  ← wastes tokens")
    print()
