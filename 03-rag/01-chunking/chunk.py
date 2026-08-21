import re

SAMPLE_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

Machine learning is a subset of artificial intelligence. It is a method of data analysis that automates analytical model building. Systems can learn from data, identify patterns, and make decisions with minimal human intervention.

Deep learning is a subset of machine learning that has networks capable of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural learning or deep neural network.

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. NLP powers search engines, translation, chatbots, and voice assistants.

Large language models (LLMs) are trained on massive text datasets using transformer architecture. They predict the next token given previous tokens, but emergence at scale produces surprisingly capable reasoning and instruction-following behavior.
""".strip()


def chunk_fixed(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def chunk_sentence(text: str, per_chunk: int = 3) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [
        " ".join(sentences[i:i + per_chunk])
        for i in range(0, len(sentences), per_chunk)
    ]


def chunk_paragraph(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def chunk_recursive(text: str, max_size: int = 300, overlap: int = 60) -> list[str]:
    if len(text) <= max_size:
        return [text.strip()] if text.strip() else []

    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                tail = current[-overlap:] if len(current) > overlap else current
                current = (tail + sep + part) if tail else part
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if c]

    return chunk_fixed(text, max_size, overlap)


if __name__ == "__main__":
    strategies = [
        ("Fixed (size=200, overlap=40)",      chunk_fixed(SAMPLE_TEXT, 200, 40)),
        ("Sentence (3 per chunk)",            chunk_sentence(SAMPLE_TEXT, 3)),
        ("Paragraph",                          chunk_paragraph(SAMPLE_TEXT)),
        ("Recursive (max=300, overlap=60)",   chunk_recursive(SAMPLE_TEXT, 300, 60)),
    ]

    for name, chunks in strategies:
        sizes = [len(c) for c in chunks]
        print(f"Strategy: {name}")
        print(f"  Chunks:   {len(chunks)}")
        print(f"  Avg size: {sum(sizes) / len(sizes):.0f} chars")
        print(f"  Min/Max:  {min(sizes)} / {max(sizes)} chars")
        print(f"  Chunk 1:  {chunks[0][:100]!r}")
        print()
