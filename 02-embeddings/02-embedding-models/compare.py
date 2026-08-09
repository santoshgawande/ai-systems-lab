import time
import math
import httpx

OLLAMA = "http://localhost:11434"

# Add or remove models based on what you've pulled: ollama list
MODELS = [
    "nomic-embed-text",
    # "mxbai-embed-large",
    # "all-minilm",
]

PAIRS = [
    ("similar",   "The stock market crashed yesterday.",         "Financial markets fell sharply on Tuesday."),
    ("different", "The stock market crashed yesterday.",         "I enjoy hiking in the mountains on weekends."),
    ("code",      "def add(a, b): return a + b",                "A function that sums two numbers."),
]


def embed(model: str, text: str) -> tuple[list[float], float]:
    start = time.time()
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": model, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"], time.time() - start


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb)


print(f"{'Model':<24} {'Dims':<6} {'Latency':<10} {'Pair':<12} {'Similarity'}")
print("-" * 70)

for model in MODELS:
    first = True
    for label, text_a, text_b in PAIRS:
        vec_a, lat_a = embed(model, text_a)
        vec_b, lat_b = embed(model, text_b)
        sim = cosine(vec_a, vec_b)
        avg_lat = (lat_a + lat_b) / 2

        model_col = model if first else ""
        dims_col = str(len(vec_a)) if first else ""
        lat_col = f"{avg_lat:.3f}s" if first else ""
        first = False

        bar = "█" * int(sim * 20)
        print(f"  {model_col:<22} {dims_col:<6} {lat_col:<10} {label:<12} {sim:.3f}  {bar}")
    print()
