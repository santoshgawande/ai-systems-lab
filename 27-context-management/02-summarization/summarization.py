"""
Summarisation-based context management: compress old conversation turns
into a running summary instead of discarding them entirely.
Preserves key information while staying within the token limit.
"""
import os
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
SUMMARY_TRIGGER_TOKENS = int(os.environ.get("SUMMARY_TRIGGER_TOKENS", "1500"))
SUMMARY_TARGET_TOKENS = int(os.environ.get("SUMMARY_TARGET_TOKENS", "300"))


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def message_tokens(msg: dict) -> int:
    return count_tokens(msg.get("content") or "") + 4


def _llm(messages: list[dict], max_tokens: int = 500) -> str:
    if OPENAI_KEY:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": max_tokens},
            timeout=30,
        )
        return r.json()["choices"][0]["message"]["content"].strip()

    if ANTHROPIC_KEY:
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "You are a helpful assistant.")
        user_msgs = [m for m in messages if m["role"] != "system"]
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": max_tokens,
                "system": sys_msg,
                "messages": user_msgs,
            },
            timeout=30,
        )
        return r.json()["content"][0]["text"].strip()

    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system")
    r = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=60,
    )
    return r.json()["response"].strip()


def summarise_turns(turns: list[dict]) -> str:
    """Ask the LLM to compress a list of conversation turns into a summary."""
    formatted = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in turns
    )
    prompt = f"""Summarise the key information from this conversation excerpt in 3-5 bullet points.
Focus on: facts established, decisions made, and user preferences revealed.
Be concise — this summary will be injected as context into future turns.

Conversation:
{formatted}

Summary:"""

    return _llm([{"role": "user", "content": prompt}], max_tokens=SUMMARY_TARGET_TOKENS)


# ─── Chatbot with summarisation ───────────────────────────────────────────────

class SummarisationChatbot:
    def __init__(self, system: str, trigger_tokens: int = SUMMARY_TRIGGER_TOKENS):
        self.system = system
        self.summary = ""               # running summary of compressed history
        self.recent_history: list[dict] = []  # recent turns (not yet compressed)
        self.trigger_tokens = trigger_tokens
        self.total_turns = 0
        self.summarisation_count = 0

    def _history_tokens(self) -> int:
        return sum(message_tokens(m) for m in self.recent_history)

    def _maybe_summarise(self) -> None:
        if self._history_tokens() < self.trigger_tokens:
            return

        # Keep last 4 messages (2 turns) as recent; compress the rest
        keep = 4
        to_compress = self.recent_history[:-keep] if len(self.recent_history) > keep else []
        if not to_compress:
            return

        print(f"  [Summarise] Compressing {len(to_compress)} messages into summary...")
        new_summary_piece = summarise_turns(to_compress)

        # Append to existing summary
        if self.summary:
            self.summary = f"{self.summary}\n\n[Later in conversation:]\n{new_summary_piece}"
        else:
            self.summary = new_summary_piece

        self.recent_history = self.recent_history[-keep:]
        self.summarisation_count += 1
        print(f"  [Summarise] Done. Summary is now {count_tokens(self.summary)} tokens.")

    def _build_messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        if self.summary:
            msgs.append({
                "role": "system",
                "content": f"[Summary of earlier conversation:]\n{self.summary}"
            })
        msgs.extend(self.recent_history)
        return msgs

    def chat(self, user_input: str) -> str:
        self.recent_history.append({"role": "user", "content": user_input})
        self._maybe_summarise()

        messages = self._build_messages()
        response = _llm(messages)
        self.recent_history.append({"role": "assistant", "content": response})
        self.total_turns += 1
        return response

    def stats(self) -> dict:
        summary_tokens = count_tokens(self.summary) if self.summary else 0
        recent_tokens = self._history_tokens()
        return {
            "turns": self.total_turns,
            "summarisations": self.summarisation_count,
            "summary_tokens": summary_tokens,
            "recent_tokens": recent_tokens,
            "total_context_tokens": summary_tokens + recent_tokens,
        }


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== SUMMARISATION CONTEXT MANAGEMENT DEMO ===\n")

# Check availability
try:
    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    llm_ok = True
except Exception:
    llm_ok = OPENAI_KEY or ANTHROPIC_KEY

if not llm_ok:
    print("No LLM available. Start Ollama or set OPENAI_API_KEY.\n")
    print("""
Summarisation pattern:

class SummarisationChatbot:
    def chat(self, user_input):
        self.history.append(user_input)

        if token_count(self.history) > TRIGGER:
            # Compress oldest turns
            old = self.history[:-4]          # all but last 2 turns
            self.summary += summarise(old)   # LLM compresses to bullets
            self.history = self.history[-4:] # keep only recent

        messages = [
            system_prompt,
            {"role": "system", "content": f"Earlier summary: {self.summary}"},
            *self.history,                   # recent turns verbatim
        ]
        return llm(messages)

Tradeoffs vs sliding window:
  Sliding window: fast, no extra LLM call, loses old info completely
  Summarisation:  extra LLM call, preserves key facts, more complex

Use summarisation when:
  - Conversation builds on earlier decisions
  - User stated preferences that should persist
  - Multi-step problem where context compounds
""")
    raise SystemExit(0)

print(f"Trigger: summarise when history exceeds {SUMMARY_TRIGGER_TOKENS} tokens\n")

bot = SummarisationChatbot(
    system="You are a technical interview coach helping prepare for software engineering interviews.",
    trigger_tokens=SUMMARY_TRIGGER_TOKENS,
)

conversation = [
    "I'm preparing for a senior backend engineer interview at a fintech company.",
    "My strongest skills are Python, PostgreSQL, and distributed systems.",
    "I have 8 years of experience and led a team of 5 engineers.",
    "What system design topics should I focus on?",
    "Can you give me a practice question about payment systems?",
    "How should I structure my answer using the STAR method?",
    "What are common mistakes candidates make in system design interviews?",
]

for msg in conversation:
    print(f"User: {msg}")
    answer = bot.chat(msg)
    stats = bot.stats()
    print(f"Bot:  {answer[:150]}...")
    print(f"[{stats['turns']} turns | recent={stats['recent_tokens']}tok | summary={stats['summary_tokens']}tok | total={stats['total_context_tokens']}tok]")
    print()

if bot.summary:
    print("─── Running summary (what was compressed) ───")
    print(bot.summary[:500])
