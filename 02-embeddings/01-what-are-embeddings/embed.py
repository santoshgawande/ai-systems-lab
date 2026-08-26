import math
import httpx

OLLAMA = "http://localhost:11434"
MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA}/api/embeddings", json={"model": MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    return dot / (norm_a * norm_b)


sentences = [
    "The cat sat on the mat.",
    "A feline rested on a rug.",
    "Dogs love to play fetch.",
    "Machine learning is a branch of AI.",
    "Deep learning uses neural networks.",
    "The Eiffel Tower is in Paris.",
]

if __name__ == "__main__":
    print(f"Embedding {len(sentences)} sentences with {MODEL}...\n")
    vectors = [(s, embed(s)) for s in sentences]
    print(f"Embedding dimensions: {len(vectors[0][1])}\n")

    print("Nearest neighbor for each sentence:")
    print("-" * 70)
    for i, (s1, v1) in enumerate(vectors):
        scores = [
            (cosine(v1, v2), s2)
            for j, (s2, v2) in enumerate(vectors)
            if i != j
        ]
        scores.sort(reverse=True)
        best_score, best_match = scores[0]
        print(f"  {s1!r}")
        print(f"  → {best_match!r}  (similarity: {best_score:.3f})\n")

    print("\nFull similarity matrix:")
    print(f"{'':>4}", end="")
    for i in range(len(sentences)):
        print(f"  S{i+1}  ", end="")
    print()
    for i, (_, v1) in enumerate(vectors):
        print(f"S{i+1} ", end="")
        for _, (_, v2) in enumerate(vectors):
            score = cosine(v1, v2)
            print(f" {score:5.2f}", end="")
        print(f"  ← {sentences[i][:30]}")
