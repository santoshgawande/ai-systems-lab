"""
Gemini long-context demo: pass large documents directly, measure accuracy by position.
Demonstrates the lost-in-the-middle problem and token counting.
Requires: GEMINI_API_KEY
"""
import os
import random

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("GEMINI_API_KEY not set. Showing long-context mechanics.\n")
    LIVE = False
else:
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    LIVE = True

MODEL = "gemini-1.5-flash"  # 1M context window; gemini-1.5-pro = higher quality

# ─── Build a synthetic long document ─────────────────────────────────────────

def build_document(num_sections: int = 60) -> tuple[str, list[dict]]:
    """
    Build a fake technical manual with num_sections sections.
    Each section has a unique fact we can query for.
    Returns (document_text, list of {section, position, key, value} facts).
    """
    sections = []
    facts = []

    topics = [
        "Installation", "Configuration", "Authentication", "Database Setup",
        "API Reference", "Networking", "Security", "Monitoring", "Logging",
        "Caching", "Performance", "Troubleshooting", "Backup", "Recovery",
        "Scaling", "Load Balancing", "SSL/TLS", "Firewall", "Updates", "Support"
    ]

    filler = (
        "This section covers important operational details for enterprise deployments. "
        "Administrators should review all subsections carefully before proceeding. "
        "Ensure all prerequisites are met and sufficient resources are allocated. "
        "Contact your system vendor for platform-specific guidance. "
    ) * 10  # ~100 words of filler per section

    for i in range(num_sections):
        topic = topics[i % len(topics)]
        section_num = i + 1
        unique_code = f"CODE-{1000 + section_num * 7}"  # unique per section
        port = 8000 + section_num

        text = f"""
## Section {section_num}: {topic}

{filler}

Configuration code for this module: {unique_code}
Default port assignment: {port}
Section checksum: {section_num * 137 % 997}

{filler}
"""
        sections.append(text)
        facts.append({
            "section": section_num,
            "position": "start" if section_num <= num_sections // 3
                        else "middle" if section_num <= 2 * num_sections // 3
                        else "end",
            "topic": topic,
            "code": unique_code,
            "port": port,
        })

    return "\n".join(sections), facts


# ─── Lost-in-the-middle test ──────────────────────────────────────────────────

def test_recall(model, document: str, facts: list[dict], n_samples: int = 9):
    """Sample facts from start/middle/end, query model, check accuracy."""
    positions = ["start", "middle", "end"]
    results = {p: {"correct": 0, "total": 0} for p in positions}

    # Pick n_samples/3 from each position
    per_pos = n_samples // 3
    sampled = []
    for pos in positions:
        pool = [f for f in facts if f["position"] == pos]
        sampled.extend(random.sample(pool, min(per_pos, len(pool))))

    prompt_prefix = f"The following is a technical manual:\n\n{document}\n\n"

    for fact in sampled:
        question = f"What is the configuration code for Section {fact['section']} ({fact['topic']})? Reply with just the code."
        full_prompt = prompt_prefix + "Question: " + question

        response = model.generate_content(full_prompt)
        answer = response.text.strip()
        correct = fact["code"] in answer

        results[fact["position"]]["total"] += 1
        results[fact["position"]]["correct"] += int(correct)

        print(f"  Section {fact['section']:3d} [{fact['position']:6s}]: "
              f"expected={fact['code']} | got={answer[:20]!r} | {'✓' if correct else '✗'}")

    print()
    for pos in positions:
        r = results[pos]
        pct = 100 * r["correct"] / r["total"] if r["total"] else 0
        print(f"  {pos:6s}: {r['correct']}/{r['total']} correct ({pct:.0f}%)")

    return results


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== GEMINI LONG CONTEXT DEMO ===\n")

if not LIVE:
    print("Gemini long context API shape:\n")
    print("""
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Count tokens first
count = model.count_tokens(large_document)
print(f"Document tokens: {count.total_tokens}")

# Pass entire document in context
response = model.generate_content([
    f"Document:\\n\\n{large_document}\\n\\nQuestion: {question}"
])
print(response.text)

# Multi-part input (separate system-like context from question)
response = model.generate_content([
    "You are a technical documentation assistant.",
    f"Documentation:\\n\\n{large_document}",
    f"Question: {question}"
])
""")
    print("Model context limits (as of 2024):")
    print("  gemini-1.5-flash: 1,000,000 tokens (cheapest)")
    print("  gemini-1.5-pro:   1,000,000 tokens (highest quality)")
    print("  gemini-1.0-pro:      32,000 tokens")
    print()
    print("Lost-in-the-middle research finding:")
    print("  Facts at the START of a long doc: ~85-90% recall")
    print("  Facts in the MIDDLE of a long doc: ~70-75% recall")
    print("  Facts at the END of a long doc:   ~85-90% recall")
    print()
    print("Mitigation strategies:")
    print("  1. Put critical info at start or end")
    print("  2. Use RAG for precision needle-in-haystack retrieval")
    print("  3. Repeat key facts in a summary at the top")
    print("  4. Use context caching for repeated queries on same document")
else:
    model = genai.GenerativeModel(MODEL)

    # Build synthetic document
    print("Building synthetic technical manual...")
    document, facts = build_document(num_sections=60)
    doc_words = len(document.split())
    print(f"Document: {len(facts)} sections, ~{doc_words:,} words\n")

    # Count tokens
    count = model.count_tokens(document)
    print(f"Token count: {count.total_tokens:,} tokens\n")

    # Simple QA test
    print("--- Simple QA on the document ---\n")
    sample_fact = facts[0]
    question = f"What is the configuration code for Section 1 (Installation)?"
    response = model.generate_content(
        f"Technical manual:\n\n{document[:50000]}\n\nQuestion: {question}\nAnswer with just the code."
    )
    print(f"Q: {question}")
    print(f"A: {response.text.strip()}")
    print(f"Expected: {sample_fact['code']}\n")

    # Lost-in-the-middle test
    print("--- Lost-in-the-middle accuracy test ---")
    print("Testing recall accuracy by document position...\n")
    test_recall(model, document, facts, n_samples=9)
