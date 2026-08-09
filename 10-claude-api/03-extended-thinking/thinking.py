"""
Claude extended thinking: allocate token budget for internal reasoning.
Like o1/o3 "thinking" but you can read Claude's reasoning in the response.
Requires: ANTHROPIC_API_KEY
"""
import os

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("ANTHROPIC_API_KEY not set.\n")
    LIVE = False
else:
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    LIVE = True

MODEL = "claude-sonnet-4-6"

HARD_PROBLEMS = [
    {
        "label": "Math reasoning",
        "prompt": "A snail climbs a 30-foot pole. During the day it climbs 3 feet. At night it slides back 2 feet. How many days does it take to reach the top?",
        "expected": "28 days (reaches top on day 28 without sliding back)",
    },
    {
        "label": "Logic puzzle",
        "prompt": "Five houses in a row are painted different colors. The English person lives in the red house. The Swede keeps dogs. The Dane drinks tea. The green house is to the left of the white house. The green house owner drinks coffee. Who drinks milk?",
        "expected": "The person in the middle house (house 3) drinks milk",
    },
    {
        "label": "Algorithm design",
        "prompt": "You have an array of integers. Find the contiguous subarray with the largest sum. What is the optimal algorithm and its time complexity?",
        "expected": "Kadane's algorithm, O(n) time, O(1) space",
    },
]

print("=== EXTENDED THINKING DEMO ===\n")

if not LIVE:
    print("API shape for extended thinking:\n")
    print("""
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8000,          # must be > budget_tokens
    thinking={
        "type": "enabled",
        "budget_tokens": 5000  # how many tokens Claude can think
    },
    messages=[{"role": "user", "content": "Hard problem here..."}]
)

# Response has multiple content blocks:
for block in response.content:
    if block.type == "thinking":
        # Claude's internal reasoning (shown to you, not charged as output)
        print(f"Thinking: {block.thinking[:200]}...")
    elif block.type == "text":
        # The actual answer
        print(f"Answer: {block.text}")

# Note: thinking content is in response but NOT included in stop_reason
# thinking tokens are billed at the same rate as input tokens
""")
    print("\nComparison (without thinking vs with thinking):\n")
    print("Without thinking: model may confidently give wrong answer to tricky problems")
    print("With thinking: model works through steps, self-corrects, gives better answer")
    print("\nBest for: complex math, logic puzzles, code review, multi-step reasoning")
    print("Skip for: simple factual questions (wasteful), creative writing (unhelpful)")
else:
    for problem in HARD_PROBLEMS:
        print(f"Problem: {problem['label']}")
        print(f"  {problem['prompt']}\n")
        print(f"  Expected: {problem['expected']}\n")

        # Without thinking
        r_direct = client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": problem["prompt"]}]
        )
        direct_answer = r_direct.content[0].text

        # With thinking
        r_thinking = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            thinking={"type": "enabled", "budget_tokens": 4000},
            messages=[{"role": "user", "content": problem["prompt"]}]
        )

        thinking_text = ""
        final_answer = ""
        for block in r_thinking.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                final_answer = block.text

        print(f"  Direct answer:  {direct_answer[:150]!r}")
        print(f"  With thinking:  {final_answer[:150]!r}")
        print(f"  Reasoning preview: {thinking_text[:200]!r}...")
        print(f"  Tokens — Direct: {r_direct.usage.output_tokens} out | "
              f"Thinking: {r_thinking.usage.output_tokens} out\n")
        print("=" * 60)
        print()
