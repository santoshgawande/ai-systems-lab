# Lab 01 — Gemini Long Context

Gemini supports up to 1M tokens in a single context window — larger than most entire codebases.

## What you learn

- How to pass large documents directly (no chunking, no vector DB)
- The "lost-in-the-middle" problem: accuracy drops for facts buried in the middle of long contexts
- When long context beats RAG, and when RAG still wins
- Token counting with `count_tokens()`
- Context caching for repeated calls on the same large document

## Run

```bash
export GEMINI_API_KEY=...
python long_context.py
```

## When to use long context vs RAG

| Scenario | Long Context | RAG |
|---|---|---|
| Single large document (< 1M tokens) | ✓ Better | — |
| Need to reason across the whole doc | ✓ Better | — |
| Many documents / knowledge base | — | ✓ Better |
| < 32K tokens needed | — | ✓ Cheaper |
| Repeated queries on same doc | ✓ With caching | ✓ |

## Lost-in-the-middle

Models are better at using information at the start and end of the context window.
Facts buried in the middle of a long document have lower recall accuracy.

```
[START]  ← high accuracy
  ...
[MIDDLE] ← accuracy drops here
  ...
[END]    ← high accuracy
```

Mitigation: put the most important info first or last. Use RAG for precision recall.

## API shape

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Pass a large document directly
response = model.generate_content([
    f"Here is a 500-page document:\n\n{document_text}\n\nQuestion: {question}"
])
print(response.text)

# Count tokens before sending
token_count = model.count_tokens(document_text)
print(f"Tokens: {token_count.total_tokens}")
```
