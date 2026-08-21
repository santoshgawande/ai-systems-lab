"""
Fine-tuning decision framework: walk through when to fine-tune vs prompt vs RAG.
Shows cost estimates, dataset sizing, and the prompting baseline test.
No API key needed — this is a decision + cost analysis tool.
"""
import json

# ─── Decision framework ───────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": "prompt_works",
        "question": "Can a well-crafted prompt (with few-shot examples) solve your task?",
        "yes": "PROMPTING",
        "no": "next"
    },
    {
        "id": "needs_knowledge",
        "question": "Does the task require specific private documents or data the model doesn't know?",
        "yes": "RAG",
        "no": "next"
    },
    {
        "id": "has_data",
        "question": "Do you have at least 50–100 high-quality labeled examples?",
        "yes": "next",
        "no": "COLLECT_DATA"
    },
    {
        "id": "stable_task",
        "question": "Will the task definition stay stable for at least 3 months?",
        "yes": "next",
        "no": "PROMPTING"
    },
    {
        "id": "consistent_format",
        "question": "Do you need EXACT consistent output format every time?",
        "yes": "FINE_TUNE",
        "no": "next"
    },
    {
        "id": "cost_pressure",
        "question": "Is the per-call cost of a strong base model (GPT-4o) too high for your volume?",
        "yes": "FINE_TUNE",
        "no": "PROMPTING"
    },
]

DECISIONS = {
    "PROMPTING": {
        "label": "Use Prompt Engineering",
        "color": "green",
        "rationale": "A good prompt is faster to iterate, cheaper to maintain, and sufficient here.",
        "action": "Write a detailed system prompt with 3-5 few-shot examples. Measure quality.",
    },
    "RAG": {
        "label": "Use RAG",
        "color": "blue",
        "rationale": "Your task needs specific knowledge. Fine-tuning won't add it reliably.",
        "action": "Build a retrieval pipeline. Chunk docs, embed, store in pgvector or Qdrant.",
    },
    "FINE_TUNE": {
        "label": "Fine-Tune",
        "color": "orange",
        "rationale": "You have the data, stable task, and clear benefit. Fine-tuning will pay off.",
        "action": "Prepare JSONL dataset, run training job, eval against baseline prompt.",
    },
    "COLLECT_DATA": {
        "label": "Collect More Data First",
        "color": "red",
        "rationale": "Too little data for fine-tuning. Use few-shot prompting while collecting.",
        "action": "Generate synthetic examples or label real cases. Target 100+ diverse samples.",
    }
}


def run_decision_tree():
    print("=== FINE-TUNING DECISION FRAMEWORK ===\n")
    print("Answer Y/N to each question:\n")

    for q in QUESTIONS:
        answer = input(f"  {q['question']} [Y/N]: ").strip().lower()
        result = q["yes"] if answer.startswith("y") else q["no"]
        if result != "next":
            decision = DECISIONS[result]
            print(f"\n  ► Recommendation: {decision['label']}")
            print(f"  Rationale: {decision['rationale']}")
            print(f"  Next step: {decision['action']}")
            return result

    # Fell through all questions
    decision = DECISIONS["FINE_TUNE"]
    print(f"\n  ► Recommendation: {decision['label']}")
    print(f"  Rationale: {decision['rationale']}")
    print(f"  Next step: {decision['action']}")
    return "FINE_TUNE"


# ─── Cost model ───────────────────────────────────────────────────────────────

def cost_analysis():
    print("\n=== COST ANALYSIS: PROMPTING vs FINE-TUNING ===\n")

    scenarios = [
        {
            "name": "Customer support classifier",
            "daily_requests": 10_000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 50,
            "few_shot_examples_tokens": 1500,
        },
        {
            "name": "Invoice data extraction",
            "daily_requests": 1_000,
            "avg_input_tokens": 800,
            "avg_output_tokens": 200,
            "few_shot_examples_tokens": 800,
        },
        {
            "name": "Code review assistant",
            "daily_requests": 500,
            "avg_input_tokens": 2000,
            "avg_output_tokens": 500,
            "few_shot_examples_tokens": 2000,
        },
    ]

    # Pricing per 1M tokens (approximate, as of 2025)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00, "training": None},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "training": None},
        "gpt-4o-mini-ft": {"input": 0.30, "output": 1.20, "training": 25.00},  # per 1M training tokens
    }

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  {scenario['daily_requests']:,} req/day × {scenario['avg_input_tokens']} in + {scenario['avg_output_tokens']} out tokens")

        daily_in = scenario["daily_requests"] * (scenario["avg_input_tokens"] + scenario["few_shot_examples_tokens"])
        daily_out = scenario["daily_requests"] * scenario["avg_output_tokens"]

        # GPT-4o with few-shot
        cost_4o = (daily_in / 1_000_000) * PRICING["gpt-4o"]["input"] + (daily_out / 1_000_000) * PRICING["gpt-4o"]["output"]

        # GPT-4o-mini with few-shot
        cost_mini = (daily_in / 1_000_000) * PRICING["gpt-4o-mini"]["input"] + (daily_out / 1_000_000) * PRICING["gpt-4o-mini"]["output"]

        # Fine-tuned mini (no few-shot examples needed in prompt)
        ft_in = scenario["daily_requests"] * scenario["avg_input_tokens"]
        cost_ft = (ft_in / 1_000_000) * PRICING["gpt-4o-mini-ft"]["input"] + (daily_out / 1_000_000) * PRICING["gpt-4o-mini-ft"]["output"]
        training_cost = 0.5  # assume 500K training tokens

        print(f"  GPT-4o (few-shot):            ${cost_4o:.2f}/day  (${cost_4o*30:.0f}/mo)")
        print(f"  GPT-4o-mini (few-shot):       ${cost_mini:.2f}/day  (${cost_mini*30:.0f}/mo)")
        print(f"  GPT-4o-mini (fine-tuned):     ${cost_ft:.2f}/day  (${cost_ft*30:.0f}/mo) + ${training_cost:.0f} one-time")
        breakeven_days = training_cost / max(cost_mini - cost_ft, 0.0001)
        print(f"  Fine-tune breaks even in:     {breakeven_days:.0f} days vs mini baseline")
        print()


# ─── Dataset sizing guide ────────────────────────────────────────────────────

def dataset_guide():
    print("=== DATASET SIZING GUIDE ===\n")

    tasks = [
        ("Binary classification", "50–200", "Low variation tasks (spam/not spam)"),
        ("Multi-class (10+ classes)", "200–500", "One diverse example per class minimum"),
        ("Named entity extraction", "200–500", "Cover all entity types + edge cases"),
        ("Style/tone consistency", "100–300", "Examples of the EXACT style you want"),
        ("Code generation", "500–2000", "Cover different languages, patterns, complexity"),
        ("Domain Q&A", "500–5000", "Depends on domain breadth"),
        ("Instruction following", "1000–10000", "Diverse instruction types"),
    ]

    for task, examples, note in tasks:
        print(f"  {task:<35} {examples:<15} {note}")

    print()
    print("Quality > Quantity: 100 perfect examples >> 1000 noisy examples")
    print("Diversity matters: cover edge cases, not just the happy path")
    print("Eval split: hold out 10-20% for evaluation (never train on these)")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("This tool helps you decide whether to fine-tune, prompt, or use RAG.\n")
    print("Running in demonstration mode (not interactive).\n")

    # Show the full decision tree without prompting
    print("=== DECISION QUESTIONS (answer these for your task) ===\n")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  {i}. {q['question']}")
        print(f"     YES → {q['yes']}")
        print(f"     NO  → {q['no']}")
        print()

    print("=== DECISIONS ===\n")
    for key, d in DECISIONS.items():
        print(f"  {key}: {d['label']}")
        print(f"    {d['rationale']}")
        print()

    cost_analysis()
    dataset_guide()

print("\n─── To run interactively: uncomment run_decision_tree() below ───")
# run_decision_tree()
