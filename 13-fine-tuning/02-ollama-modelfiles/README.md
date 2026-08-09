# Lab 02 — Ollama Modelfiles

Create custom Ollama models by writing a Modelfile — like a Dockerfile for LLMs.

## What you learn

- Modelfile syntax: FROM, SYSTEM, PARAMETER, TEMPLATE, MESSAGE
- Baking a system prompt into the model (no need to send it every call)
- Tuning temperature, top_p, top_k, stop sequences per model
- Creating domain-specific personas (support bot, code reviewer, etc.)
- Quantization selection: which GGUF quant for your hardware

## Run

```bash
python modelfiles.py
# This script generates Modelfiles and shows you how to build/test them
# Requires: ollama running at localhost:11434
```

## Modelfile syntax

```dockerfile
# Base model (any model in your Ollama library)
FROM llama3.2

# Bake in a system prompt
SYSTEM """
You are a senior Python code reviewer. Focus on:
1. Security vulnerabilities
2. Performance issues
3. Pythonic style (PEP 8)
Always give specific line-number references.
"""

# Model parameters
PARAMETER temperature 0.3      # Lower = more consistent
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096         # Context window
PARAMETER stop "<|eot_id|>"    # Stop sequences

# Optional: pre-loaded few-shot examples
MESSAGE user "Review: x = 1; y = 2; print(x+y)"
MESSAGE assistant "Line 1: Combine into a single expression for clarity. No issues found."
```

## Build and use

```bash
# Create the model
ollama create my-code-reviewer -f Modelfile

# Use it
ollama run my-code-reviewer "Review this function: ..."

# Via API
curl http://localhost:11434/api/generate -d '{
  "model": "my-code-reviewer",
  "prompt": "Review: def f(x): return eval(x)"
}'

# List your custom models
ollama list
```

## GGUF quantization guide

| Quant | Size (7B) | RAM needed | Quality |
|---|---|---|---|
| Q2_K | ~3 GB | 4 GB | Low |
| Q4_K_M | ~4.5 GB | 6 GB | Good (sweet spot) |
| Q5_K_M | ~5.5 GB | 8 GB | Very good |
| Q8_0 | ~7.5 GB | 10 GB | Near full |
| F16 | ~14 GB | 16 GB | Full precision |

Mac Studio M4 Max 64GB → Q8_0 for 7B, Q4_K_M for 30B+ models.
