# 01 — LLM APIs

Raw API calls to local Ollama models. No frameworks, no abstractions — just HTTP.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Ollama running at `http://localhost:11434`.
Check with: `curl http://localhost:11434/api/tags`

## Labs

| Dir | What you learn | Run |
|---|---|---|
| `01-hello-ollama/` | Raw request/response, message format | `python hello.py` |
| `02-streaming/` | Token streaming (how ChatGPT works) | `python stream.py` |
| `03-multi-model-compare/` | Query N models in parallel, compare outputs | `python compare.py "your prompt"` |
| `04-model-router/` | Route task to best model automatically | `python router.py "your prompt"` |
| `05-benchmark/` | Measure latency and tokens/sec per model | `python benchmark.py` |
| `06-vision/` | Image Q&A, OCR, multi-model vision compare, batch processing | `python image_qa.py img.jpg "what is this?"` |

## Models used

| Model | Tag |
|---|---|
| General / balanced | `llama3.3:70b` |
| Reasoning / math | `deepseek-r1` |
| Code generation | `qwen2.5-coder:32b` |
| Fast / lightweight | `phi4` |
| Embeddings | `nomic-embed-text` |

Change model tags in each script if your pulled models differ.
Run `ollama list` to see what's available locally.
