"""
Gemini grounding with Google Search: compare grounded vs ungrounded responses.
Shows how to read grounding metadata (search queries, sources).
Requires: GEMINI_API_KEY (grounding requires 1.5 Flash or Pro)
"""
import os

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("GEMINI_API_KEY not set. Showing grounding mechanics.\n")
    LIVE = False
else:
    import google.generativeai as genai
    from google.generativeai import types as gtypes
    genai.configure(api_key=API_KEY)
    LIVE = True

MODEL = "gemini-1.5-flash"

# Questions where grounding helps (recent, factual, current events)
GROUNDING_HELPS = [
    "What are the latest AI model releases from major labs in 2025?",
    "What is the current version of Python?",
    "What happened in AI news this week?",
]

# Questions where grounding doesn't change much (timeless / reasoning)
GROUNDING_NEUTRAL = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is the time complexity of quicksort?",
    "Write a Python function to reverse a linked list.",
]


def ask_ungrounded(model_plain, question: str) -> str:
    response = model_plain.generate_content(question)
    return response.text.strip()


def ask_grounded(model_grounded, question: str) -> tuple[str, list[str], list[str]]:
    """Returns (answer, search_queries_used, source_urls)."""
    response = model_grounded.generate_content(question)
    answer = response.text.strip()

    search_queries = []
    sources = []

    candidates = response.candidates
    if candidates and candidates[0].grounding_metadata:
        meta = candidates[0].grounding_metadata
        search_queries = list(meta.web_search_queries or [])
        for chunk in (meta.grounding_chunks or []):
            if hasattr(chunk, "web") and chunk.web:
                title = getattr(chunk.web, "title", "") or ""
                uri = getattr(chunk.web, "uri", "") or ""
                if uri:
                    sources.append(f"{title} — {uri}" if title else uri)

    return answer, search_queries, sources


def compare(model_plain, model_grounded, question: str):
    print(f"Q: {question}\n")

    ungrounded = ask_ungrounded(model_plain, question)
    grounded, queries, sources = ask_grounded(model_grounded, question)

    print(f"Without grounding:\n  {ungrounded[:300]}\n")
    print(f"With grounding:\n  {grounded[:300]}\n")

    if queries:
        print(f"  Searches used: {queries}")
    if sources:
        print(f"  Sources: {len(sources)} cited")
        for s in sources[:3]:
            print(f"    • {s[:100]}")

    print("-" * 70 + "\n")


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== GEMINI GROUNDING DEMO ===\n")

if not LIVE:
    print("Grounding API shape:\n")
    print("""
import google.generativeai as genai
from google.generativeai import types as gtypes

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Model WITHOUT grounding (default)
model_plain = genai.GenerativeModel("gemini-1.5-flash")

# Model WITH Google Search grounding
model_grounded = genai.GenerativeModel(
    "gemini-1.5-flash",
    tools=[gtypes.Tool(google_search_retrieval=gtypes.GoogleSearchRetrieval())]
)

# Use grounded model exactly like normal
response = model_grounded.generate_content("What are the latest GPT models?")
print(response.text)

# Read grounding metadata
meta = response.candidates[0].grounding_metadata
print("Search queries:", meta.web_search_queries)
for chunk in meta.grounding_chunks:
    if chunk.web:
        print(f"Source: {chunk.web.title} — {chunk.web.uri}")

# Grounding supports
# also supports "dynamic retrieval" with a threshold:
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    tools=[gtypes.Tool(google_search_retrieval=gtypes.GoogleSearchRetrieval(
        dynamic_retrieval_config=gtypes.DynamicRetrievalConfig(
            mode=gtypes.DynamicRetrievalConfig.Mode.MODE_DYNAMIC,
            dynamic_threshold=0.3  # only ground when confidence < 0.3
        )
    ))]
)
""")
    print("When grounding helps:")
    print("  ✓ Current events (last week, today, latest releases)")
    print("  ✓ Version numbers, release dates, recent stats")
    print("  ✓ Stock prices, sports scores, weather")
    print()
    print("When grounding doesn't help:")
    print("  - Timeless concepts (algorithms, math, history > 1 year)")
    print("  - Code generation (not information retrieval)")
    print("  - Reasoning tasks (logic, analysis)")
    print()
    print("Important constraints:")
    print("  - Cannot combine grounding + JSON response_format")
    print("  - Adds ~1-2s latency for live search")
    print("  - Not all API tiers support grounding")
else:
    # Create both model variants
    model_plain = genai.GenerativeModel(MODEL)

    try:
        model_grounded = genai.GenerativeModel(
            MODEL,
            tools=[gtypes.Tool(google_search_retrieval=gtypes.GoogleSearchRetrieval())]
        )
        print("Grounding enabled.\n")
    except Exception as e:
        print(f"Could not enable grounding: {e}")
        print("Falling back to comparison without grounding metadata.\n")
        model_grounded = model_plain

    # Section 1: Questions where grounding helps
    print("=== Grounding helps: recent/factual questions ===\n")
    for q in GROUNDING_HELPS[:2]:
        compare(model_plain, model_grounded, q)

    # Section 2: Questions where grounding is neutral
    print("=== Grounding neutral: timeless/reasoning questions ===\n")
    for q in GROUNDING_NEUTRAL[:2]:
        compare(model_plain, model_grounded, q)

    print("Takeaway:")
    print("  Enable grounding when your prompt references current events or recent facts.")
    print("  Skip grounding for reasoning, code, and math — it adds latency with no benefit.")
