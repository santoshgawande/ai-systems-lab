import json
import math
import httpx
from pathlib import Path
from datetime import datetime

OLLAMA = "http://localhost:11434"
MODEL = "llama3.3:70b"
EMBED_MODEL = "nomic-embed-text"
MEMORY_FILE = Path("/tmp/agent_memory.json")

SYSTEM_BASE = """You are a helpful assistant with persistent memory.
When the user shares personal info (name, preferences, facts), acknowledge it.
Reference prior knowledge when relevant."""


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na and nb else 0.0


class Memory:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"episodes": [], "facts": []}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def add_episode(self, role: str, content: str):
        self.data["episodes"].append({"role": role, "content": content, "ts": datetime.now().isoformat()})
        self.save()

    def add_fact(self, fact: str):
        self.data["facts"].append({"text": fact, "embedding": embed(fact), "ts": datetime.now().isoformat()})
        self.save()

    def recent_turns(self, n: int = 6) -> list[dict]:
        """Sliding window: last N messages."""
        return [{"role": e["role"], "content": e["content"]} for e in self.data["episodes"][-n:]]

    def relevant_facts(self, query: str, top_k: int = 3, min_score: float = 0.5) -> list[str]:
        """Semantic retrieval: facts most similar to current query."""
        if not self.data["facts"]:
            return []
        q_vec = embed(query)
        scored = sorted(
            ((cosine(q_vec, f["embedding"]), f["text"]) for f in self.data["facts"]),
            reverse=True,
        )
        return [text for score, text in scored[:top_k] if score >= min_score]


def extract_facts(user_msg: str, assistant_msg: str) -> list[str]:
    """Ask the LLM to extract memorable facts about the user."""
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Extract memorable facts about the user from this exchange. Return a JSON array of strings, or [] if nothing notable."},
            {"role": "user", "content": f"User: {user_msg}\nAssistant: {assistant_msg}"},
        ],
        "stream": False,
    }, timeout=30)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        start, end = content.find("["), content.rfind("]") + 1
        return json.loads(content[start:end]) if start >= 0 else []
    except Exception:
        return []


mem = Memory(MEMORY_FILE)
print(f"Memory: {len(mem.data['episodes'])} episodes, {len(mem.data['facts'])} facts  (stored at {MEMORY_FILE})")
print("Commands: 'memory' to list facts, 'quit' to exit\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "memory":
        print("\nStored facts:")
        for f in mem.data["facts"]:
            print(f"  - {f['text']}")
        print()
        continue

    facts = mem.relevant_facts(user_input)
    system = SYSTEM_BASE
    if facts:
        system += "\n\nRelevant memories:\n" + "\n".join(f"- {f}" for f in facts)

    messages = [{"role": "system", "content": system}] + mem.recent_turns(6) + [{"role": "user", "content": user_input}]

    r = httpx.post(f"{OLLAMA}/api/chat", json={"model": MODEL, "messages": messages, "stream": False}, timeout=60)
    r.raise_for_status()
    response = r.json()["message"]["content"]

    print(f"Assistant: {response}\n")

    mem.add_episode("user", user_input)
    mem.add_episode("assistant", response)

    new_facts = extract_facts(user_input, response)
    for fact in new_facts:
        mem.add_fact(fact)
        print(f"  [Memory saved] {fact}")
