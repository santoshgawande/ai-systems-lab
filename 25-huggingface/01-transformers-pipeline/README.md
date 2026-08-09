# Lab 01 — Transformers Pipeline

One-liner inference for any NLP task — no cloud API needed.

## What you learn

- `pipeline()` for sentiment, NER, summarisation, zero-shot classification, Q&A
- Device selection: CPU / CUDA GPU / Apple Silicon MPS
- Model caching: HuggingFace downloads to `~/.cache/huggingface/` on first run
- When to use small task-specific models vs large LLMs

## Run

```bash
pip install transformers torch
python pipeline_demo.py          # all tasks (slow, downloads ~1-2GB)
python pipeline_demo.py --quick  # sentiment + NER only (~200MB)
```

## Key patterns

```python
from transformers import pipeline

# Simple (downloads default model)
clf = pipeline("sentiment-analysis")
print(clf("I love this!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Specific model
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")

# Apple Silicon GPU
pipe = pipeline("text-generation", model="gpt2", device="mps")

# CUDA GPU
pipe = pipeline("text-generation", model="gpt2", device=0)
```

## Task vs LLM: when to use which

| Task | Small pipeline model | LLM |
|------|---------------------|-----|
| Sentiment | DistilBERT (66MB) — 10ms | GPT-4o — 500ms, $0.001 |
| NER | BERT-NER (433MB) — 20ms | GPT-4o — 400ms, $0.002 |
| Summarisation | BART-CNN (1.6GB) — 300ms | Claude — 800ms, $0.003 |
| Complex reasoning | Not suitable | LLM only |

Rule of thumb: for single-task inference at scale, a fine-tuned small model wins on cost and latency.
