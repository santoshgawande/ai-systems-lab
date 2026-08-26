from __future__ import annotations
"""
Pairwise Preference Dataset Pipeline & Length-Bias Normalizer.

A major failure mode in RLHF/DPO is Verbosity Exploitation (length bias):
Models learn that writing long, fluffy answers trick human evaluators and reward models
into higher ratings regardless of accuracy.

This pipeline:
1. Ingests raw candidate generation pairs (A vs B).
2. Computes pairwise quality scores using rule-based and LLM heuristics.
3. Detects and penalizes length bias when longer answers are unnecessarily verbose.
4. Formats clean JSONL datasets ready for DPO / SimPO fine-tuning.
"""
from typing import Any, Dict, List, Optional, Tuple
import dataclasses
import json


@dataclasses.dataclass
class RawCandidatePair:
    prompt: str
    response_a: str
    response_b: str
    score_a: float
    score_b: float


@dataclasses.dataclass
class DPOFormattedPair:
    prompt: str
    chosen: str
    rejected: str
    margin: float
    length_ratio: float


class PreferenceDataPipeline:
    """
    Constructs clean, unbiased pairwise preference datasets for DPO training.
    """
    def __init__(self, min_margin: float = 0.15, max_length_ratio: float = 3.0):
        self.min_margin = min_margin
        self.max_length_ratio = max_length_ratio

    def process_pair(self, pair: RawCandidatePair) -> Optional[DPOFormattedPair]:
        """
        Determines chosen vs rejected, penalizes length bias, and validates score margin.
        """
        len_a = len(pair.response_a.split())
        len_b = len(pair.response_b.split())
        
        # Calculate length ratio
        ratio = max(len_a, len_b) / max(1, min(len_a, len_b))
        
        # If response is > 3x longer with only negligible quality gain, penalize verbosity
        adjusted_score_a = pair.score_a
        adjusted_score_b = pair.score_b

        if len_a > len_b * 2.0 and (pair.score_a - pair.score_b) < 0.1:
            adjusted_score_a -= 0.15  # Penalize fluff
        elif len_b > len_a * 2.0 and (pair.score_b - pair.score_a) < 0.1:
            adjusted_score_b -= 0.15

        margin = abs(adjusted_score_a - adjusted_score_b)
        if margin < self.min_margin:
            # Pair is too ambiguous (scores too close to reliably distinguish preference)
            return None

        if adjusted_score_a > adjusted_score_b:
            chosen = pair.response_a
            rejected = pair.response_b
        else:
            chosen = pair.response_b
            rejected = pair.response_a

        return DPOFormattedPair(
            prompt=pair.prompt,
            chosen=chosen,
            rejected=rejected,
            margin=margin,
            length_ratio=ratio
        )

    def export_dpo_jsonl(self, pairs: List[RawCandidatePair]) -> List[Dict[str, str]]:
        dataset = []
        for p in pairs:
            formatted = self.process_pair(p)
            if formatted:
                dataset.append({
                    "prompt": formatted.prompt,
                    "chosen": formatted.chosen,
                    "rejected": formatted.rejected
                })
        return dataset


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 📊 PREFERENCE DATASET & LENGTH-BIAS PIPELINE ===\n")

    pipeline = PreferenceDataPipeline(min_margin=0.2, max_length_ratio=3.0)

    raw_data = [
        RawCandidatePair(
            prompt="What is the capital of India?",
            response_a="The capital of India is New Delhi.",
            response_b="India, officially the Republic of India, is a country in South Asia with many historical cities including Mumbai, Kolkata, Chennai, Bangalore, and its capital city New Delhi.",
            score_a=0.95,  # Concise & direct
            score_b=0.90   # Fluffy
        ),
        RawCandidatePair(
            prompt="Is 17 a prime number?",
            response_a="Yes, 17 is prime.",
            response_b="No, 17 is divisible by 3.",
            score_a=1.0,
            score_b=0.0
        ),
        RawCandidatePair(
            prompt="What is the color of the sky?",
            response_a="The sky is blue.",
            response_b="The sky appears blue due to Rayleigh scattering.",
            score_a=0.85,
            score_b=0.88   # Margin too small (< 0.2)
        )
    ]

    print(f"Processing {len(raw_data)} candidate candidate pairs...")
    jsonl_output = pipeline.export_dpo_jsonl(raw_data)
    
    print(f"\nGenerated {len(jsonl_output)} clean DPO preference pairs:")
    for i, item in enumerate(jsonl_output, 1):
        print(f"\n[Pair {i}] Prompt: {item['prompt']}")
        print(f"  (+) Chosen:   {item['chosen']}")
        print(f"  (-) Rejected: {item['rejected']}")

    print("\nTakeaway: High quality alignment datasets filter out ambiguous ties and penalize verbosity gaming.")
