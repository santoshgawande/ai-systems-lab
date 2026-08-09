"""
Sliding window context management: discard oldest messages when approaching the token limit.
Demonstrates token counting, window sizing, and the eviction strategy.
"""
import os
import re

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
CONTEXT_LIMIT = int(os.environ.get("CONTEXT_LIMIT", "4096"))
RESERVE_FOR_OUTPUT = 512

# ─── Token counting ───────────────────────────────────────────────────────────

def count_tokens_approx(text: str) -> int:
    """~4 chars per token (rough estimate, no tiktoken needed)."""
    return max(1, len(text) // 4)


def count_tokens_tiktoken(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return count_tokens_approx(text)


count_tokens = count_tokens_tiktoken  # use best available


def message_tokens(msg: dict) -> int:
    content = msg.get("content") or ""
    role = msg.get("role", "")
    # OpenAI adds ~4 tokens per message for role/format overhead
    return count_tokens(content) + count_tokens(role) + 4


def conversation_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


# ─── Sliding window ───────────────────────────────────────────────────────────

def apply_sliding_window(
    messages: list[dict],
    max_tokens: int,
    reserve_system: bool = True,
) -> tuple[list[dict], int]:
    """
    Evict oldest non-system messages until messages fit in max_tokens.
    Returns (trimmed_messages, evicted_count).
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    evicted = 0
    while non_system:
        total = conversation_tokens(system_msgs + non_system)
        if total <= max_tokens:
            break
        # Evict oldest non-system message
        non_system.pop(0)
        evicted += 1

    return system_msgs + non_system, evicted


# ─── Chatbot with sliding window ─────────────────────────────────────────────

class SlidingWindowChatbot:
    def __init__(self, system: str, max_context_tokens: int = CONTEXT_LIMIT):
        self.system_msg = {"role": "system", "content": system}
        self.history: list[dict] = []
        self.max_context = max_context_tokens
        self.max_for_history = max_context_tokens - RESERVE_FOR_OUTPUT

    def _build_messages(self) -> list[dict]:
        all_msgs = [self.system_msg] + self.history
        trimmed, evicted = apply_sliding_window(all_msgs, self.max_for_history)
        if evicted:
            print(f"  [Window] Evicted {evicted} message(s) to stay within {self.max_for_history} tokens")
        return trimmed

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        messages = self._build_messages()

        response = _call_llm(messages)
        self.history.append({"role": "assistant", "content": response})
        return response

    def stats(self) -> dict:
        total = conversation_tokens([self.system_msg] + self.history)
        return {
            "turns": len(self.history) // 2,
            "total_tokens": total,
            "history_messages": len(self.history),
            "limit": self.max_context,
            "utilisation_pct": round(total / self.max_context * 100, 1),
        }


def _call_llm(messages: list[dict]) -> str:
    import httpx
    if OPENAI_KEY:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": RESERVE_FOR_OUTPUT},
            timeout=30,
        )
        return r.json()["choices"][0]["message"]["content"].strip()

    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

    if ANTHROPIC_KEY:
        sys_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": RESERVE_FOR_OUTPUT,
                "system": sys_content,
                "messages": user_msgs,
            },
            timeout=30,
        )
        return r.json()["content"][0]["text"].strip()

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    r = httpx.post(
        f"{os.environ.get('OLLAMA_BASE', 'http://localhost:11434')}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=60,
    )
    return r.json()["response"].strip()


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== SLIDING WINDOW CONTEXT DEMO ===\n")

# Show token counting
sample_messages = [
    {"role": "system", "content": "You are a helpful assistant for a software company."},
    {"role": "user", "content": "How do I set up a PostgreSQL connection pool in Python?"},
    {"role": "assistant", "content": "Use SQLAlchemy with create_engine and pool_size parameter..."},
    {"role": "user", "content": "What's the difference between pool_size and max_overflow?"},
    {"role": "assistant", "content": "pool_size is the number of permanent connections. max_overflow adds temporary ones..."},
]

print("Token counting example:")
print(f"  Approx (char/4): {count_tokens_approx(sample_messages[1]['content'])} tokens")
try:
    import tiktoken
    print(f"  Tiktoken exact:  {count_tokens_tiktoken(sample_messages[1]['content'])} tokens")
except ImportError:
    print(f"  (tiktoken not installed — using approx counting)")

total = conversation_tokens(sample_messages)
print(f"  Full conversation: {total} tokens\n")

# Simulate window eviction
print("Sliding window simulation (max_tokens=200):")
trimmed, evicted = apply_sliding_window(sample_messages, max_tokens=200)
print(f"  Original: {len(sample_messages)} messages, {conversation_tokens(sample_messages)} tokens")
print(f"  After trim: {len(trimmed)} messages, {conversation_tokens(trimmed)} tokens")
print(f"  Evicted: {evicted} oldest messages")
print()

# Live chatbot demo
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
import httpx
try:
    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    llm_ok = True
except Exception:
    llm_ok = OPENAI_KEY or bool(os.environ.get("ANTHROPIC_API_KEY"))

if llm_ok:
    print("Live chatbot with sliding window (limit=800 tokens for demo):\n")
    bot = SlidingWindowChatbot(
        system="You are a concise technical assistant. Keep answers under 3 sentences.",
        max_context_tokens=800,
    )
    questions = [
        "What is a database index?",
        "What types of indexes exist in PostgreSQL?",
        "When should I avoid using an index?",
        "How do I check if my query is using an index?",
        "What is a covering index?",
    ]
    for q in questions:
        print(f"  User: {q}")
        answer = bot.chat(q)
        stats = bot.stats()
        print(f"  Bot:  {answer[:120]}...")
        print(f"  Stats: {stats['turns']} turns, {stats['total_tokens']} tokens ({stats['utilisation_pct']}% of limit)")
        print()
else:
    print("No LLM available for live demo. Start Ollama or set API keys.")
    print("\nSliding window rule:")
    print("  Keep evicting oldest messages until conversation fits in token limit.")
    print("  Always preserve: system message + most recent N turns.")
    print("  Trade-off: lose early context but stay within cost/limit bounds.")
