"""
Model quantisation: run large models in less memory.
Covers INT8, INT4 (bitsandbytes), and GPTQ concepts.
NOTE: bitsandbytes requires Linux or Windows with CUDA.
      On Mac, use Ollama instead (handles quantisation internally).
"""
import os
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import bitsandbytes  # noqa: F401
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


# ─── Memory math ──────────────────────────────────────────────────────────────

def model_memory_table() -> None:
    print("Memory required by model size and precision:\n")
    models = [
        ("7B",  7),
        ("13B", 13),
        ("30B", 30),
        ("70B", 70),
    ]
    precisions = [
        ("FP32 (32-bit)", 4),
        ("FP16 / BF16",   2),
        ("INT8",          1),
        ("INT4 (NF4)",    0.5),
    ]
    header = f"  {'Model':<12}" + "".join(f"{name:<22}" for name, _ in precisions)
    print(header)
    print("  " + "─" * (12 + 22 * len(precisions)))
    for model_name, params_b in models:
        row = f"  {model_name:<12}"
        for _, bytes_per_param in precisions:
            gb = params_b * 1e9 * bytes_per_param / 1e9
            row += f"{gb:.1f} GB{'':<17}"
        print(row)
    print()
    print("  GPU VRAM available (reference):")
    gpus = [
        ("RTX 3090 / 4090", "24 GB"),
        ("RTX 4080",        "16 GB"),
        ("Mac M4 Max",      "64 GB unified (MPS)"),
        ("A100 80GB",       "80 GB"),
    ]
    for name, vram in gpus:
        print(f"    {name:<25} {vram}")


def quantisation_methods() -> None:
    print("""
Quantisation methods:

1. BitsAndBytes INT8 (LLM.int8())
   - 8-bit weights + activations
   - ~50% memory reduction vs FP16
   - ~10% slower inference
   - Linux + CUDA only

   from transformers import BitsAndBytesConfig
   config = BitsAndBytesConfig(load_in_8bit=True)
   model = AutoModelForCausalLM.from_pretrained(name, quantization_config=config)

2. BitsAndBytes NF4 / QLoRA (4-bit)
   - 4-bit NF4 weights (normal float 4)
   - ~75% memory reduction vs FP16
   - NestedQuant (double quantisation) saves another ~0.4 bits/param
   - Used for QLoRA fine-tuning

   config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",            # NF4 > INT4 for LLMs
       bnb_4bit_compute_dtype=torch.bfloat16, # compute in BF16
       bnb_4bit_use_double_quant=True,        # nested quantisation
   )
   model = AutoModelForCausalLM.from_pretrained(name, quantization_config=config)

3. GPTQ (post-training quantisation)
   - 4-bit weights, calibrated on sample data
   - Faster inference than bitsandbytes
   - Pre-quantised models on HuggingFace (TheBloke/)
   - pip install auto-gptq

   from auto_gptq import AutoGPTQForCausalLM
   model = AutoGPTQForCausalLM.from_quantized("TheBloke/Llama-2-7B-GPTQ")

4. GGUF (llama.cpp format) — used by Ollama
   - Cross-platform (CPU + GPU), best for Mac
   - Q4_K_M, Q5_K_M, Q8_0 quantisation levels
   - ollama pull llama3.2  ← handles this automatically

   Level guide:
     Q4_K_M  — smallest, good quality (~4.1 bits/param)
     Q5_K_M  — slightly better quality (~5.1 bits/param)
     Q8_0    — near-lossless (~8.5 bits/param)
     F16     — no quantisation (full FP16)
""")


def accuracy_tradeoffs() -> None:
    print("""
Accuracy tradeoffs (approximate):

  FP16   baseline  — full quality, most memory
  INT8   ~99%      — nearly lossless, good for most tasks
  NF4    ~96-98%   — slight degradation, suitable for generation
  INT4   ~93-95%   — noticeable on knowledge-heavy tasks
  INT2   ~70-80%   — significant degradation, experimental

Rule of thumb:
  - Inference serving: use INT8 or GPTQ INT4 for 2x speedup + memory savings
  - Fine-tuning: use QLoRA (NF4 base + FP16 adapters) to fit in consumer GPU
  - Mac / edge: use Ollama with Q4_K_M for best accuracy/size balance

Perplexity benchmark (lower = better; Llama-2-7B example):
  FP16:   5.47
  INT8:   5.50  (+0.5%)
  GPTQ4:  5.68  (+3.8%)
  GGUF4:  5.72  (+4.6%)
""")


def live_demo() -> None:
    """Load a tiny model in INT8 to show the API works."""
    if not (HF_AVAILABLE and BNB_AVAILABLE and TORCH_AVAILABLE):
        return

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available — skipping live quantisation demo (requires CUDA GPU)")
        print("On Mac: use Ollama (handles GGUF quantisation internally)")
        return

    MODEL = "facebook/opt-125m"  # tiny model for demo
    print(f"\nLoading {MODEL} in INT8...")

    try:
        config = BitsAndBytesConfig(load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=config,
            device_map="auto",
        )

        prompt = "The key benefit of quantisation is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        import time
        t0 = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        elapsed = time.perf_counter() - t0

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        print(f"Time: {elapsed*1000:.0f}ms")

        # Memory stats
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak GPU memory: {mem:.0f} MB")

    except Exception as e:
        print(f"Error: {e}")


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== MODEL QUANTISATION DEMO ===\n")

model_memory_table()
print()
quantisation_methods()
accuracy_tradeoffs()
live_demo()

print("─── Mac / homelab recommendation ───")
print("  Mac Studio M4 Max 64GB:")
print("    → Use Ollama with Q4_K_M GGUF models")
print("    → ollama pull llama3.2, mistral, codellama")
print("    → No bitsandbytes needed — Ollama handles quantisation")
print()
print("  Linux GPU server (for fine-tuning):")
print("    → QLoRA with NF4 base model + PEFT adapters")
print("    → pip install bitsandbytes accelerate peft")
print("    → 7B model fits in single RTX 3090 (24GB)")
