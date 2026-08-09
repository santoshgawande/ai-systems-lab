# Lab 02 — Model Quantisation

Fit large models in less memory. Run 7B models on consumer hardware.

## What you learn

- Memory math: FP32 → FP16 → INT8 → INT4 → memory requirements per model size
- BitsAndBytes INT8 and NF4 (QLoRA) quantisation
- GPTQ: pre-quantised models from HuggingFace Hub
- GGUF (Ollama): cross-platform quantisation for Mac/CPU

## Run

```bash
pip install transformers torch
# bitsandbytes requires Linux + CUDA:
pip install bitsandbytes accelerate
python quantization.py
```

**Mac users**: this lab is primarily educational. Use Ollama (which handles GGUF quantisation automatically) for local inference on Apple Silicon.

## Memory reference

| Model | FP16 | INT8 | INT4 (NF4) |
|-------|------|------|------------|
| 7B  | 14 GB | 7 GB | 4 GB |
| 13B | 26 GB | 13 GB | 7 GB |
| 70B | 140 GB | 70 GB | 35 GB |

## Quantisation APIs

```python
# INT8 (bitsandbytes, CUDA only)
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=config)

# NF4 / QLoRA (4-bit, CUDA only)
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# GGUF on Mac via Ollama (recommended for homelab)
# ollama pull llama3.2    # Q4_K_M by default
```

## Choosing a quantisation level

| Level | Quality loss | Use case |
|-------|-------------|----------|
| INT8 | ~0.5% | Production inference servers |
| GPTQ INT4 | ~4% | Constrained GPU memory |
| GGUF Q4_K_M | ~5% | Mac / CPU inference (Ollama) |
| NF4 (QLoRA) | ~3% | Fine-tuning base model |
