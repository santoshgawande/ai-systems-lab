"""
HuggingFace transformers pipeline: one-liner inference for NLP tasks.
Covers: text generation, sentiment, NER, summarisation, zero-shot classification, Q&A.
"""
import sys
import time

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ─── Task catalogue ───────────────────────────────────────────────────────────

TASKS = [
    {
        "name": "Sentiment Analysis",
        "task": "sentiment-analysis",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "inputs": [
            "This product is absolutely fantastic!",
            "Terrible experience. Broke after two days.",
            "It's okay I guess, not great not terrible.",
        ],
    },
    {
        "name": "Named Entity Recognition",
        "task": "ner",
        "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "inputs": [
            "Apple CEO Tim Cook announced new products in Cupertino, California.",
            "Elon Musk founded SpaceX and Tesla, both headquartered in the United States.",
        ],
        "kwargs": {"aggregation_strategy": "simple"},
    },
    {
        "name": "Zero-Shot Classification",
        "task": "zero-shot-classification",
        "model": "facebook/bart-large-mnli",
        "inputs": [
            "The stock market dropped 5% today after disappointing earnings reports.",
        ],
        "kwargs": {"candidate_labels": ["finance", "sports", "technology", "politics"]},
    },
    {
        "name": "Summarisation",
        "task": "summarization",
        "model": "facebook/bart-large-cnn",
        "inputs": [
            """The transformer architecture, introduced in the seminal paper "Attention Is All You Need"
            in 2017, revolutionised natural language processing. Unlike previous recurrent models,
            transformers use self-attention mechanisms to process all tokens in parallel, enabling
            much faster training on modern hardware. The architecture consists of encoder and decoder
            stacks, each containing multi-head attention layers and feed-forward networks. BERT, GPT,
            and their successors all build on this foundation."""
        ],
        "kwargs": {"max_length": 60, "min_length": 20, "do_sample": False},
    },
    {
        "name": "Question Answering",
        "task": "question-answering",
        "model": "distilbert-base-cased-distilled-squad",
        "inputs": None,  # handled separately
        "qa_pairs": [
            {
                "question": "Who created the transformer architecture?",
                "context": "The transformer architecture was introduced by researchers at Google Brain and Google Research in 2017 in the paper 'Attention Is All You Need' by Vaswani et al.",
            },
        ],
    },
]

# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== HUGGINGFACE TRANSFORMERS PIPELINE DEMO ===\n")

if not HF_AVAILABLE:
    print("transformers not installed. pip install transformers torch\n")
    print("""
The pipeline() API — one-liner inference:

from transformers import pipeline

# Sentiment
clf = pipeline("sentiment-analysis")
print(clf("I love this product!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
gen = pipeline("text-generation", model="gpt2")
print(gen("The future of AI is", max_new_tokens=30))

# NER (named entity recognition)
ner = pipeline("ner", aggregation_strategy="simple")
print(ner("Elon Musk founded Tesla in California"))
# [{'entity_group': 'PER', 'word': 'Elon Musk', ...}]

# Zero-shot classification (no training needed)
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zsc("Breaking: markets fall 5%", candidate_labels=["finance", "sports"]))

# Summarisation
summ = pipeline("summarization", model="facebook/bart-large-cnn")
print(summ(long_text, max_length=60))

# Question answering (extractive)
qa = pipeline("question-answering")
print(qa(question="Who made it?", context="Tesla was founded by Elon Musk."))

# Device selection:
pipeline("sentiment-analysis", device=0)         # GPU 0 (CUDA)
pipeline("sentiment-analysis", device="mps")     # Apple Silicon GPU
pipeline("sentiment-analysis", device=-1)        # CPU (default)
""")
    raise SystemExit(0)

# Detect best device
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = 0  # first GPU
else:
    DEVICE = -1  # CPU
print(f"Device: {DEVICE}\n")

# Run tasks
QUICK_MODE = "--quick" in sys.argv or len(sys.argv) > 1

# In quick mode, only run the smallest models
tasks_to_run = TASKS[:2] if QUICK_MODE else TASKS

for task_def in tasks_to_run:
    print(f"{'─'*60}")
    print(f"Task: {task_def['name']}  (model: {task_def['model']})")
    print("Loading model...")

    try:
        kwargs = task_def.get("kwargs", {})
        pipe = pipeline(task_def["task"], model=task_def["model"], device=DEVICE)

        start = time.perf_counter()

        if task_def["task"] == "question-answering":
            for pair in task_def["qa_pairs"]:
                result = pipe(question=pair["question"], context=pair["context"])
                print(f"  Q: {pair['question']}")
                print(f"  A: {result['answer']}  (score={result['score']:.3f})")
        else:
            for inp in task_def["inputs"]:
                result = pipe(inp, **kwargs)
                print(f"  Input: {inp[:70]}")
                if isinstance(result, list):
                    for item in result[:3]:
                        if isinstance(item, dict):
                            # NER, classification
                            label = item.get("entity_group") or item.get("label") or item.get("labels", [None])[0]
                            score = item.get("score", item.get("scores", [None])[0])
                            word = item.get("word", "")
                            print(f"    → {label}  {f'({word})' if word else ''}  score={score:.3f}")
                        elif isinstance(item, str):
                            print(f"    → {item[:100]}")
                else:
                    # Summarisation returns dict directly
                    if "summary_text" in result:
                        print(f"  Summary: {result['summary_text']}")
                print()

        elapsed = time.perf_counter() - start
        print(f"  Inference time: {elapsed*1000:.0f}ms")

    except Exception as e:
        print(f"  Error: {e}")
        print("  (Model may need to download on first run — be patient)")
    print()

print("─── Pipeline task reference ───")
tasks_ref = [
    ("text-generation",         "GPT-style text completion"),
    ("sentiment-analysis",      "Positive/negative classification"),
    ("ner",                     "Named entity recognition"),
    ("zero-shot-classification","Classify without labelled training data"),
    ("summarization",           "Abstractive text summarisation"),
    ("question-answering",      "Extractive QA from context"),
    ("translation_en_to_fr",    "Machine translation"),
    ("fill-mask",               "BERT-style masked token prediction"),
    ("image-classification",    "Vision: classify an image"),
    ("automatic-speech-recognition", "Whisper-style STT"),
]
for task, desc in tasks_ref:
    print(f"  {task:<40} {desc}")
