# Lab 03 — Gemini Grounding with Google Search

Gemini can ground responses in real-time Google Search results, reducing hallucination on time-sensitive or factual queries.

## What you learn

- How to enable Google Search grounding in a Gemini request
- How grounding changes response accuracy on recent events
- Reading `grounding_metadata` — search queries used, sources cited
- Grounded vs ungrounded comparison on factual questions
- When grounding helps (recent facts) vs when it doesn't (general reasoning)

## Run

```bash
export GEMINI_API_KEY=...
python grounding.py
```

## API shape

```python
import google.generativeai as genai
from google.generativeai.types import Tool, GoogleSearchRetrieval

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    tools=[Tool(google_search_retrieval=GoogleSearchRetrieval())]
)

response = model.generate_content("What happened in tech news today?")
print(response.text)

# Grounding metadata
if response.candidates[0].grounding_metadata:
    meta = response.candidates[0].grounding_metadata
    for query in meta.web_search_queries:
        print(f"Search: {query}")
    for chunk in meta.grounding_chunks:
        print(f"Source: {chunk.web.title} — {chunk.web.uri}")
```

## Grounding vs RAG vs long context

| Approach | Best for | Latency | Cost |
|---|---|---|---|
| No grounding | Reasoning, coding, writing | Low | Low |
| Grounding (Search) | Current events, recent facts | Medium | Medium |
| Long context | Deep doc analysis | Medium | Medium |
| RAG | Large private knowledge base | Medium | Low |

## Limitations

- Grounding requires `gemini-1.5-flash` or `gemini-1.5-pro` (not 1.0)
- Cannot use grounding + JSON mode simultaneously
- Adds ~1-2s latency (live search)
- Not available for free-tier API in some regions
