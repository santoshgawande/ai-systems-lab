"""
OpenAI Batch API: submit JSONL, poll completion, retrieve results at 50% cost.
Demonstrates bulk classification with error handling and cost comparison.
Requires: OPENAI_API_KEY
"""
import os
import json
import time
import io

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("OPENAI_API_KEY not set. Showing Batch API mechanics.\n")
    LIVE = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    LIVE = True

MODEL = "gpt-4o-mini"

# ─── Sample dataset: product reviews to classify ─────────────────────────────

REVIEWS = [
    ("rev-001", "Absolutely love this product! Best purchase I've made all year."),
    ("rev-002", "Arrived broken. Complete waste of money. Would give zero stars if I could."),
    ("rev-003", "It's okay. Does what it says. Nothing special but no complaints either."),
    ("rev-004", "Customer service was incredible when I had an issue. Resolved in minutes!"),
    ("rev-005", "Overpriced for what you get. Found a better alternative for half the price."),
    ("rev-006", "Using it every day for 6 months. Still works great. Highly recommend."),
    ("rev-007", "Packaging was nice but the product itself is disappointing."),
    ("rev-008", "Five stars for delivery speed. Product is exactly as described."),
    ("rev-009", "I'm on my third return. Quality control seems nonexistent."),
    ("rev-010", "Exactly what I needed for my home office setup. Perfect size and quality."),
]

CLASSIFY_SYSTEM = "Classify the sentiment. Respond with exactly one word: positive, negative, or neutral."


def build_batch_jsonl(reviews: list[tuple[str, str]]) -> bytes:
    lines = []
    for review_id, text in reviews:
        request = {
            "custom_id": review_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 10,
            }
        }
        lines.append(json.dumps(request))
    return "\n".join(lines).encode()


def poll_batch(batch_id: str, timeout: int = 120) -> object:
    deadline = time.time() + timeout
    while time.time() < deadline:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        print(f"  Status: {status} | total={counts.total} completed={counts.completed} failed={counts.failed}")

        if status == "completed":
            return batch
        elif status in ("failed", "cancelled", "expired"):
            print(f"  Batch ended: {status}")
            return batch

        time.sleep(5)
    print("  Timeout waiting for batch.")
    return None


def parse_results(output_file_id: str) -> dict[str, str]:
    content = client.files.content(output_file_id)
    results = {}
    for line in content.text.strip().split("\n"):
        item = json.loads(line)
        custom_id = item["custom_id"]
        if item.get("error"):
            results[custom_id] = f"ERROR: {item['error']['message']}"
        else:
            results[custom_id] = item["response"]["body"]["choices"][0]["message"]["content"].strip()
    return results


def estimate_cost(n_requests: int, avg_input_tokens: int, avg_output_tokens: int) -> dict:
    standard_in = 0.15 / 1_000_000
    standard_out = 0.60 / 1_000_000
    batch_in = 0.075 / 1_000_000
    batch_out = 0.30 / 1_000_000

    standard = n_requests * (avg_input_tokens * standard_in + avg_output_tokens * standard_out)
    batch = n_requests * (avg_input_tokens * batch_in + avg_output_tokens * batch_out)
    return {"standard": standard, "batch": batch, "savings": standard - batch}


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== OPENAI BATCH API DEMO ===\n")
print(f"Dataset: {len(REVIEWS)} reviews to classify\n")

if not LIVE:
    print("Batch API flow:\n")
    print("""
# Step 1: Build JSONL and upload
jsonl_bytes = build_batch_jsonl(reviews)
file = client.files.create(file=("batch.jsonl", jsonl_bytes), purpose="batch")

# Step 2: Create batch
batch = client.batches.create(
    input_file_id=file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

# Step 3: Poll until done
while batch.status not in ("completed", "failed"):
    time.sleep(30)
    batch = client.batches.retrieve(batch.id)
    print(f"Status: {batch.status}")

# Step 4: Retrieve results
results = parse_results(batch.output_file_id)
for custom_id, sentiment in results.items():
    print(f"{custom_id}: {sentiment}")

# Step 5: Cancel if needed (before completion)
client.batches.cancel(batch.id)
""")
    print("Cost analysis for 10,000 reviews (avg 100 input + 5 output tokens):")
    costs = estimate_cost(10_000, 100, 5)
    print(f"  Standard API: ${costs['standard']:.4f}")
    print(f"  Batch API:    ${costs['batch']:.4f}")
    print(f"  Savings:      ${costs['savings']:.4f} ({costs['savings']/costs['standard']*100:.0f}%)")
else:
    # Build JSONL
    print("Building batch JSONL...")
    jsonl = build_batch_jsonl(REVIEWS)
    print(f"  {len(REVIEWS)} requests, {len(jsonl)} bytes\n")

    # Upload
    print("Uploading batch file...")
    batch_file = client.files.create(
        file=("batch_reviews.jsonl", io.BytesIO(jsonl)),
        purpose="batch"
    )
    print(f"  File ID: {batch_file.id}\n")

    # Create batch
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}\n")

    # Poll (with timeout — demo may complete quickly for small batches)
    print("Polling batch status (max 2 min for demo)...")
    completed_batch = poll_batch(batch.id, timeout=120)

    if completed_batch and completed_batch.status == "completed":
        print(f"\nBatch completed!")
        results = parse_results(completed_batch.output_file_id)

        print("\nResults:")
        for review_id, text in REVIEWS:
            sentiment = results.get(review_id, "MISSING")
            print(f"  {review_id}: {sentiment:<10}  {text[:60]!r}")

        # Cost comparison
        counts = completed_batch.request_counts
        costs = estimate_cost(counts.completed, 80, 5)
        print(f"\nCost comparison ({counts.completed} requests):")
        print(f"  Standard API: ${costs['standard']:.6f}")
        print(f"  Batch API:    ${costs['batch']:.6f}")
        print(f"  Saved:        ${costs['savings']:.6f} ({50:.0f}%)")

        # Cleanup
        client.files.delete(batch_file.id)
        if completed_batch.output_file_id:
            client.files.delete(completed_batch.output_file_id)
    else:
        print("\nBatch still running (normal — up to 24h).")
        print(f"Check status: client.batches.retrieve('{batch.id}')")
        print(f"Cancel:       client.batches.cancel('{batch.id}')")
