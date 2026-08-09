"""
Anthropic Message Batches API: bulk Claude requests at 50% cost.
Demonstrates batch creation, polling, result streaming.
Requires: ANTHROPIC_API_KEY
"""
import os
import time

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("ANTHROPIC_API_KEY not set. Showing Message Batches API mechanics.\n")
    LIVE = False
else:
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    LIVE = True

MODEL = "claude-haiku-4-5-20251001"   # cheapest Claude — ideal for batch

# ─── Sample workload: classify 10 code snippets ──────────────────────────────

CODE_SNIPPETS = [
    ("snippet-001", "def add(a, b): return a + b"),
    ("snippet-002", "query = f\"SELECT * FROM users WHERE id = {user_id}\""),
    ("snippet-003", "password = 'admin123'  # TODO: move to env var"),
    ("snippet-004", "result = eval(user_input)"),
    ("snippet-005", "with open(filename, 'r') as f: return f.read()"),
    ("snippet-006", "subprocess.run(user_command, shell=True)"),
    ("snippet-007", "os.remove(path)"),
    ("snippet-008", "requests.get(url, verify=False)"),
    ("snippet-009", "hash = md5(password.encode()).hexdigest()"),
    ("snippet-010", "return json.loads(response.text)"),
]

SECURITY_SYSTEM = (
    "You are a security code reviewer. "
    "Respond with JSON only: {\"risk\": \"high|medium|low|none\", \"issue\": \"brief description or null\"}"
)


def poll_batch(batch_id: str, timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"  Status: {status} | request_counts={batch.request_counts}")

        if status == "ended":
            return batch
        time.sleep(5)

    print("  Timeout — batch still processing.")
    return None


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== ANTHROPIC MESSAGE BATCHES DEMO ===\n")
print(f"Dataset: {len(CODE_SNIPPETS)} code snippets for security analysis\n")

if not LIVE:
    print("Message Batches API shape:\n")
    print("""
import anthropic
client = anthropic.Anthropic()

# Create batch — list of MessageCreateParamsNonStreaming
batch = client.messages.batches.create(
    requests=[
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            custom_id="req-001",
            params={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 100,
                "system": "You are a code reviewer.",
                "messages": [{"role": "user", "content": code_snippet}],
            }
        )
        for custom_id, code_snippet in CODE_SNIPPETS
    ]
)
print(f"Batch ID: {batch.id}")  # msgbatch_...

# Poll until status == "ended"
while batch.processing_status != "ended":
    time.sleep(10)
    batch = client.messages.batches.retrieve(batch.id)

# Stream results (no need to download a file — results streamed via SDK)
for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        text = result.result.message.content[0].text
        print(f"{result.custom_id}: {text}")
    elif result.result.type == "errored":
        print(f"{result.custom_id}: ERROR — {result.result.error.error.message}")
    elif result.result.type == "expired":
        print(f"{result.custom_id}: EXPIRED")

# Cancel (before it ends)
client.messages.batches.cancel(batch.id)

# List all batches
for b in client.messages.batches.list():
    print(b.id, b.processing_status, b.request_counts)
""")
    print("Cost: same 50% discount as OpenAI batch")
    print("Models: any Claude model (Haiku is cheapest for bulk tasks)")
    print("Limit: 10,000 requests per batch, 100MB max input")
else:
    # Build batch requests
    requests = [
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            custom_id=custom_id,
            params={
                "model": MODEL,
                "max_tokens": 80,
                "system": SECURITY_SYSTEM,
                "messages": [{"role": "user", "content": f"Review this code:\n{snippet}"}],
            }
        )
        for custom_id, snippet in CODE_SNIPPETS
    ]

    print("Creating batch...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")
    print(f"  Status:   {batch.processing_status}\n")

    print("Polling batch status (max 2 min)...")
    completed = poll_batch(batch.id, timeout=120)

    if completed and completed.processing_status == "ended":
        results_map = {}
        for result in client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                results_map[result.custom_id] = result.result.message.content[0].text
            elif result.result.type == "errored":
                results_map[result.custom_id] = f"ERROR: {result.result.error.error.message}"

        print(f"\nResults ({len(results_map)}/{len(CODE_SNIPPETS)} returned):\n")
        import json
        for custom_id, snippet in CODE_SNIPPETS:
            raw = results_map.get(custom_id, "MISSING")
            try:
                data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
                risk = data.get("risk", "?")
                issue = data.get("issue") or "none"
            except Exception:
                risk, issue = "?", raw[:50]
            marker = "⚠" if risk == "high" else ("·" if risk == "none" else "!")
            print(f"  {marker} [{risk:<6}] {snippet:<50}  {issue}")
    else:
        print("\nBatch still processing (normal for large batches — up to 24h).")
        print(f"Check later: client.messages.batches.retrieve('{batch.id}')")
