# Lab 03 — LoRA / QLoRA Concepts

Fine-tune a 7B+ model on a laptop by only updating 0.1% of parameters.

## What you learn

- What LoRA is: low-rank adapter matrices injected into attention layers
- Why only 0.1% of parameters need updating for task adaptation
- QLoRA: quantize the base model to 4-bit, train adapters in float16
- How to prepare a fine-tuning dataset (JSONL chat format)
- Tools: HuggingFace PEFT, Unsloth (2x faster LoRA training)

## Run

```bash
python lora_concepts.py
# Conceptual demo — shows math + dataset prep without GPU required
```

## LoRA in 60 seconds

A transformer attention layer has weight matrix **W** (d × d, e.g. 4096×4096 = 16M params).

LoRA adds: **W' = W + BA**
- **B** is d × r (e.g. 4096 × 8)
- **A** is r × d (e.g. 8 × 4096)
- r is the "rank" (typically 8–64)
- Total new params: 2 × d × r = 65,536 (vs 16M in W)

Only B and A are trained. W is frozen.

## Memory comparison (7B model, batch=1)

| Approach | GPU RAM | Training time |
|---|---|---|
| Full fine-tune (bf16) | ~56 GB | 1x baseline |
| LoRA (r=64, bf16) | ~28 GB | 0.6x |
| QLoRA (4-bit base, r=64) | ~10 GB | 0.4x |
| Unsloth QLoRA | ~8 GB | 0.2x (2x speedup) |

## Rank selection guide

| Task | Rank (r) | Why |
|---|---|---|
| Style/format (simple) | 8–16 | Low rank captures stylistic patterns |
| Domain knowledge | 32–64 | More capacity for new knowledge |
| Instruction following | 16–32 | Medium complexity |
| Function calling | 64–128 | High-precision output format |

## Dataset format (JSONL, Alpaca-style)

```json
{"instruction": "Classify this email as spam or not spam.", "input": "You won $1M! Click here.", "output": "spam"}
{"instruction": "Classify this email as spam or not spam.", "input": "Meeting at 3pm today.", "output": "not spam"}
```

Or chat format:
```json
{"messages": [{"role": "user", "content": "Classify: 'You won $1M'"}, {"role": "assistant", "content": "spam"}]}
```
