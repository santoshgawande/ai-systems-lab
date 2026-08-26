from __future__ import annotations
"""
Speculative Sampling Engine (Leviathan et al., DeepMind 2023).

Accelerates autoregressive LLM decoding without changing the output distribution.
A lightweight Draft Model (e.g. 1B params) speculatively drafts K tokens.
The heavyweight Target Model (e.g. 70B params) evaluates all K tokens in a SINGLE forward pass.

Rejection sampling rule:
  For each drafted token x_i:
    Accept if r < min(1, q(x_i) / p(x_i)) where r ~ Uniform(0, 1)
    If rejected at position j:
      Resample x_j from the residual distribution: norm(max(0, q(x) - p(x)))
      Discard remaining draft tokens x_{j+1} ... x_K
"""
import random
import math
from typing import Dict, List, Tuple, Optional


def normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalize a probability dictionary so sum(p) == 1.0."""
    total = sum(dist.values())
    if total <= 0:
        return {k: 1.0 / len(dist) for k in dist}
    return {k: v / total for k, v in dist.items()}


def sample_from_distribution(dist: Dict[str, float], rng: Optional[random.Random] = None) -> str:
    """Sample a token according to the probability distribution."""
    r_val = (rng or random).random()
    cumulative = 0.0
    for token, prob in dist.items():
        cumulative += prob
        if r_val <= cumulative:
            return token
    return list(dist.keys())[-1]


def compute_residual_distribution(
    target_dist: Dict[str, float],
    draft_dist: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute residual distribution when target model rejects draft token:
        residual(x) = max(0, q(x) - p(x)) / sum(max(0, q(x') - p(x')))
    """
    all_tokens = set(target_dist.keys()) | set(draft_dist.keys())
    diff = {}
    for tok in all_tokens:
        q = target_dist.get(tok, 0.0)
        p = draft_dist.get(tok, 0.0)
        diff[tok] = max(0.0, q - p)
    
    return normalize_distribution(diff)


class SpeculativeSamplingEngine:
    """
    Simulates speculative sampling between a Draft Model and a Target Model.
    """
    def __init__(self, gamma: int = 4, seed: Optional[int] = 42):
        self.gamma = gamma  # Lookahead draft tokens count K
        self.rng = random.Random(seed)
        self.total_drafted = 0
        self.total_accepted = 0
        self.target_forward_passes = 0
        self.generated_tokens: List[str] = []

    def verify_draft_sequence(
        self,
        draft_tokens: List[str],
        draft_distributions: List[Dict[str, float]],
        target_distributions: List[Dict[str, float]],
    ) -> Tuple[List[str], int]:
        """
        Runs exact rejection sampling across the drafted sequence.
        Returns:
            (accepted_and_corrected_tokens, accepted_count)
        """
        accepted: List[str] = []
        num_accepted = 0

        for i, token in enumerate(draft_tokens):
            self.total_drafted += 1
            p_draft = draft_distributions[i].get(token, 1e-9)
            q_target = target_distributions[i].get(token, 0.0)

            # Acceptance probability: min(1, q / p)
            acceptance_prob = min(1.0, q_target / p_draft if p_draft > 0 else 0.0)
            roll = self.rng.random()

            if roll < acceptance_prob:
                accepted.append(token)
                num_accepted += 1
                self.total_accepted += 1
            else:
                # Token rejected! Sample replacement token from residual distribution
                residual = compute_residual_distribution(
                    target_distributions[i],
                    draft_distributions[i]
                )
                replacement_token = sample_from_distribution(residual, self.rng)
                accepted.append(replacement_token)
                # Discard remaining speculative tokens in this batch
                break

        # If all gamma tokens accepted, target model generates one additional bonus token
        if num_accepted == len(draft_tokens) and len(target_distributions) > len(draft_tokens):
            bonus_dist = target_distributions[-1]
            bonus_token = sample_from_distribution(bonus_dist, self.rng)
            accepted.append(bonus_token)

        self.target_forward_passes += 1
        return accepted, num_accepted

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def estimated_speedup(self) -> float:
        """
        Theoretical speedup formula:
            Speedup = (1 - (1-alpha)^(gamma+1)) / ((1-alpha) * (1 + gamma * (1-alpha)))
        Simplified approximation:
            Tokens per target forward pass = len(generated) / target_forward_passes
        """
        if self.target_forward_passes == 0:
            return 1.0
        return (len(self.generated_tokens) or (self.total_accepted + self.target_forward_passes)) / self.target_forward_passes


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🚀 SPECULATIVE SAMPLING ENGINE (Leviathan et al.) ===\n")

    vocab = ["def", "solve", "(", "n", ":", "int", ")", "->", "bool", ":", "\n", "    return", "n", ">", "0"]
    engine = SpeculativeSamplingEngine(gamma=4, seed=123)

    # Simulated distributions for 5 step iterations
    demo_steps = [
        ("Step 1 (High Alignment)", ["def", "solve", "(", "n"], [
            {"def": 0.9, "solve": 0.05, "class": 0.05},
            {"solve": 0.85, "calculate": 0.1, "main": 0.05},
            {"(": 0.95, "[": 0.05},
            {"n": 0.9, "x": 0.1}
        ], [
            {"def": 0.92, "solve": 0.04, "class": 0.04},
            {"solve": 0.88, "calculate": 0.08, "main": 0.04},
            {"(": 0.98, "[": 0.02},
            {"n": 0.92, "x": 0.08},
            {":": 0.85, ",": 0.15}  # Bonus token distribution
        ]),
        ("Step 2 (Moderate Alignment with 1 rejection)", [":", "int", ")", "->"], [
            {":": 0.8, "->": 0.2},
            {"int": 0.7, "str": 0.3},
            {")": 0.9, ",": 0.1},
            {"->": 0.85, ":": 0.15}
        ], [
            {":": 0.85, "->": 0.15},
            {"int": 0.1, "Any": 0.8, "str": 0.1},  # Target disagrees strongly here!
            {")": 0.95, ",": 0.05},
            {"->": 0.9, ":": 0.1}
        ])
    ]

    all_emitted = []
    for title, draft_toks, p_dists, q_dists in demo_steps:
        print(f"--- {title} ---")
        print(f"Draft Model proposed K={len(draft_toks)} tokens: {draft_toks}")
        accepted, n_acc = engine.verify_draft_sequence(draft_toks, p_dists, q_dists)
        all_emitted.extend(accepted)
        engine.generated_tokens.extend(accepted)
        print(f"Target Model verification result: {accepted} (Accepted {n_acc}/{len(draft_toks)})")
        print(f"Current Cumulative Acceptance Rate: {engine.acceptance_rate:.1%}")
        print(f"Current Empirical Speedup: {engine.estimated_speedup:.2f}x\n")

    print(f"Final Generated Tokens: {' '.join(all_emitted)}")
    print(f"Total Drafted: {engine.total_drafted}, Total Target Steps: {engine.target_forward_passes}")
    print(f"Key Takeaway: Speculative Decoding matches target probability distribution with zero loss in output quality!")
