# Section 25 — HuggingFace Transformers

Run open-source models locally with the transformers pipeline API and reduce memory with quantisation.

## What you learn

- `pipeline()` — one-liner inference for NLP, vision, and audio tasks
- Model loading, tokenisation, and inference without cloud APIs
- Quantisation: INT8 and INT4 to fit large models in limited VRAM
- When local HuggingFace models beat cloud APIs (latency, cost, privacy)

## Labs

| Lab | What it covers |
|---|---|
| 01-transformers-pipeline | pipeline() API, NLP tasks, CPU vs GPU inference |
| 02-quantization | bitsandbytes INT4, GPTQ, model size comparison |

## Setup

```bash
pip install -r requirements.txt
```

## Hardware notes (your homelab)

- Mac Studio M4 Max 64GB — use `device="mps"` for GPU acceleration on Apple Silicon
- Most 7B models fit in 16GB GPU memory (FP16)
- With INT4 quantisation, 7B models fit in ~4-6GB
