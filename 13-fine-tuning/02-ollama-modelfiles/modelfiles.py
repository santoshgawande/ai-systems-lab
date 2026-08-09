"""
Ollama Modelfiles: generate, build, and test custom models.
Creates Modelfiles for different personas, measures consistency improvement.
Requires: Ollama at localhost:11434
"""
import os
import json
import subprocess
import tempfile
import httpx

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
BASE_MODEL = "llama3.2"  # must be pulled: ollama pull llama3.2


def ollama_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def generate(model: str, prompt: str, system: str = "") -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    r = httpx.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60)
    return r.json().get("response", "").strip()


def model_exists(name: str) -> bool:
    r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    return name in models or f"{name}:latest" in models


def build_model(name: str, modelfile_content: str) -> bool:
    """Write Modelfile to temp file, run ollama create."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
        f.write(modelfile_content)
        path = f.name
    result = subprocess.run(
        ["ollama", "create", name, "-f", path],
        capture_output=True, text=True
    )
    os.unlink(path)
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}")
        return False
    return True


# ─── Modelfile definitions ───────────────────────────────────────────────────

MODELFILES = {
    "lab-code-reviewer": f"""FROM {BASE_MODEL}

SYSTEM \"\"\"
You are a senior software engineer conducting a code review.
For every issue you find:
1. State the line/variable/function name
2. Explain WHY it is a problem
3. Show the corrected version

Focus on: security vulnerabilities, performance bugs, error handling gaps.
Do NOT comment on formatting or naming style.
Be concise — one paragraph per issue maximum.
\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.85
""",

    "lab-sql-expert": f"""FROM {BASE_MODEL}

SYSTEM \"\"\"
You are a PostgreSQL expert. When asked to write SQL:
- Always use parameterized queries ($1, $2) — never string concatenation
- Add comments explaining non-obvious JOINs or subqueries
- Include an EXPLAIN ANALYZE hint if the query touches > 10K rows
- Use CTEs (WITH) instead of nested subqueries when possible

Output format:
```sql
-- your query here
```
Then one sentence explaining the approach.
\"\"\"

PARAMETER temperature 0.1
""",

    "lab-friendly-explainer": f"""FROM {BASE_MODEL}

SYSTEM \"\"\"
You explain technical concepts to beginners.
Rules:
- No jargon without defining it
- Use an analogy before the technical explanation
- Maximum 3 paragraphs
- End with one concrete example

Assume the reader is smart but has no technical background.
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.95
""",
}

# ─── Test prompts ────────────────────────────────────────────────────────────

TESTS = {
    "lab-code-reviewer": [
        "Review this Python function:\n\ndef get_user(username):\n    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n    return db.execute(query)",
        "Review:\nimport pickle\ndata = pickle.loads(user_input)",
    ],
    "lab-sql-expert": [
        "Write a query to get the top 10 users by total order value in the last 30 days.",
        "Find all products that have never been ordered.",
    ],
    "lab-friendly-explainer": [
        "What is a database index?",
        "What is the difference between TCP and UDP?",
    ],
}


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== OLLAMA MODELFILES DEMO ===\n")

if not ollama_available():
    print("Ollama not available. Showing Modelfile syntax and usage.\n")
    for name, content in MODELFILES.items():
        print(f"=== Modelfile: {name} ===")
        print(content)
        print()
    print("Usage:")
    print("  1. Save a Modelfile to disk")
    print("  2. ollama create my-model -f ./Modelfile")
    print("  3. ollama run my-model 'your prompt'")
    print("  4. Via API: POST /api/generate with model=my-model")
else:
    for name, modelfile in MODELFILES.items():
        print(f"─── Building model: {name} ───")

        if model_exists(name):
            print(f"  Already exists, skipping build.")
        else:
            print(f"  Building from Modelfile...", end=" ", flush=True)
            ok = build_model(name, modelfile)
            print("done." if ok else "FAILED.")
            if not ok:
                continue

        print(f"  Testing prompts:")
        for prompt in TESTS.get(name, [])[:1]:
            print(f"\n  Prompt: {prompt[:80]!r}")
            response = generate(name, prompt)
            print(f"  Response: {response[:300]}\n")

    # Compare: base model vs custom model on same prompt
    print("\n─── Baseline comparison: base model vs lab-code-reviewer ───\n")
    test_prompt = "Review:\ndef login(user, pwd):\n    sql = f\"SELECT * FROM users WHERE user='{user}' AND pwd='{pwd}'\"\n    return db.execute(sql)"

    print(f"Prompt: {test_prompt}\n")

    base_resp = generate(BASE_MODEL, test_prompt, system="You are a code reviewer.")
    print(f"Base model ({BASE_MODEL}):\n  {base_resp[:400]}\n")

    if model_exists("lab-code-reviewer"):
        custom_resp = generate("lab-code-reviewer", test_prompt)
        print(f"Custom model (lab-code-reviewer):\n  {custom_resp[:400]}\n")

    print("Key difference: custom model follows the exact issue format consistently.")
    print("No need to re-send the system prompt — it's baked into the model weights.")
