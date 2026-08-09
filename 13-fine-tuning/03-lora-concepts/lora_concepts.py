"""
LoRA / QLoRA concepts: visualize the math, show parameter counts, prepare datasets.
No GPU required — this is a conceptual + data prep demo.
"""
import json
import math
import random

# ─── LoRA math visualization ─────────────────────────────────────────────────

def lora_param_count(d: int, r: int, num_layers: int = 32, targets: int = 4) -> dict:
    """
    Calculate trainable parameters for LoRA.
    targets: number of attention matrices updated per layer (q, k, v, o = 4)
    """
    full_params = d * d * targets * num_layers
    lora_params = 2 * d * r * targets * num_layers  # A + B matrices
    ratio = lora_params / full_params
    return {
        "full_finetune_params": full_params,
        "lora_params": lora_params,
        "ratio": ratio,
        "full_finetune_gb_bf16": full_params * 2 / 1e9,
        "lora_only_gb_bf16": lora_params * 2 / 1e9,
    }


print("=== LORA CONCEPTS ===\n")

print("── Parameter count comparison ──\n")
configs = [
    ("7B model (Llama 3.2)", 4096, 32, 32, 4),
    ("13B model", 5120, 40, 32, 4),
    ("70B model (Llama 3.3)", 8192, 80, 64, 4),
]

for name, d, r_rank, layers, targets in configs:
    c = lora_param_count(d, r_rank, layers, targets)
    print(f"  {name} (rank={r_rank})")
    print(f"    Full fine-tune: {c['full_finetune_params']/1e9:.1f}B params  ({c['full_finetune_gb_bf16']:.1f} GB bf16)")
    print(f"    LoRA adapters:  {c['lora_params']/1e6:.1f}M params  ({c['lora_only_gb_bf16']:.2f} GB bf16)")
    print(f"    Ratio: {c['ratio']*100:.3f}% of full model\n")


# ─── Rank vs quality trade-off ───────────────────────────────────────────────

print("── Rank selection guide ──\n")
rank_guide = [
    (4,   "Classification, simple formatting", "~0.01% params", "Minimal"),
    (8,   "Style/tone adjustment", "~0.02% params", "Low"),
    (16,  "Instruction following, moderate tasks", "~0.04% params", "Medium"),
    (32,  "Domain knowledge, entity extraction", "~0.08% params", "Good"),
    (64,  "Complex reasoning, function calling", "~0.16% params", "High"),
    (128, "Near full fine-tune quality", "~0.32% params", "Very high"),
]

print(f"  {'Rank':<6} {'Use case':<45} {'Params':<15} {'Capacity'}")
print(f"  {'-'*90}")
for rank, use_case, params, capacity in rank_guide:
    print(f"  {rank:<6} {use_case:<45} {params:<15} {capacity}")
print()


# ─── Dataset preparation ─────────────────────────────────────────────────────

print("── Dataset preparation ──\n")

# Simulate a small training dataset (sentiment classification)
raw_data = [
    ("This product exceeded my expectations! Absolutely love it.", "positive"),
    ("Terrible quality. Broke after one day. Complete waste of money.", "negative"),
    ("It's okay. Does the job but nothing special.", "neutral"),
    ("Amazing customer service! They resolved my issue instantly.", "positive"),
    ("The worst experience I've ever had with any company.", "negative"),
    ("Pretty good value for the price. Would buy again.", "positive"),
    ("Arrived damaged. Packaging was inadequate.", "negative"),
    ("Exactly what I ordered. Fast shipping.", "neutral"),
    ("Life-changing product! Can't imagine living without it.", "positive"),
    ("Instructions were confusing and product is mediocre.", "negative"),
]


def to_chat_format(text: str, label: str) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Classify the sentiment of the following text. Respond with exactly one word: positive, negative, or neutral."
            },
            {
                "role": "user",
                "content": text
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    }


def to_alpaca_format(text: str, label: str) -> dict:
    return {
        "instruction": "Classify the sentiment. Respond with exactly: positive, negative, or neutral.",
        "input": text,
        "output": label
    }


# Split train/eval
random.shuffle(raw_data)
split = int(len(raw_data) * 0.8)
train_data = raw_data[:split]
eval_data = raw_data[split:]

# Write JSONL files
chat_train = [to_chat_format(t, l) for t, l in train_data]
chat_eval = [to_chat_format(t, l) for t, l in eval_data]

import tempfile, os
tmpdir = tempfile.mkdtemp()

train_path = os.path.join(tmpdir, "train.jsonl")
eval_path = os.path.join(tmpdir, "eval.jsonl")

with open(train_path, "w") as f:
    for ex in chat_train:
        f.write(json.dumps(ex) + "\n")

with open(eval_path, "w") as f:
    for ex in chat_eval:
        f.write(json.dumps(ex) + "\n")

print(f"  Train: {len(chat_train)} examples → {train_path}")
print(f"  Eval:  {len(chat_eval)} examples → {eval_path}")
print()
print("  First training example (chat format):")
print(json.dumps(chat_train[0], indent=4))
print()


# ─── QLoRA memory estimate ───────────────────────────────────────────────────

print("── QLoRA memory breakdown (7B model) ──\n")
components = [
    ("Base model weights (4-bit NF4)", 7e9 * 0.5 / 1e9, "GB"),  # 4-bit = 0.5 bytes/param
    ("LoRA adapters (bf16, r=64)", 0.067, "GB"),
    ("Optimizer states (AdamW, bf16)", 0.134, "GB"),
    ("Activations + gradients", 1.5, "GB"),
    ("TOTAL", None, "GB"),
]
total = sum(v for _, v, _ in components if v is not None)
components[-1] = ("TOTAL", total, "GB")

for label, val, unit in components:
    bar = "█" * int(val * 5) if val else ""
    print(f"  {label:<40} {val:.2f} {unit}  {bar}")

print()
print("  Compare: Full fine-tune (bf16) = ~56 GB → needs A100 80GB")
print("  QLoRA = ~10 GB → fits on RTX 3090 / 4090 (24 GB VRAM)")
print()


# ─── Training script skeleton ────────────────────────────────────────────────

print("── Training script (Unsloth / PEFT) ──\n")
print("""
# pip install unsloth peft transformers datasets trl

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # QLoRA base
    max_seq_length=2048,
    load_in_4bit=True,        # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                     # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=load_dataset("json", data_files="train.jsonl")["train"],
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
    ),
)
trainer.train()

# Save adapter only (tiny file — push to HuggingFace Hub)
model.save_pretrained("my-sentiment-lora")  # ~50 MB vs 16 GB full model
""")
