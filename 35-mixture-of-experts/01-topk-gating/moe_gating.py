from __future__ import annotations
"""
Sparse Mixture of Experts (MoE) Top-K Gating Router (Mixtral 8x7B & DeepSeek-V3).

MoE decouples total model capacity (e.g. 671B parameters) from inference compute cost
(e.g. only 37B active parameters per token).

Architecture:
1. Gating Network: Linear projection mapping token embedding -> E expert logits.
2. Top-K Selection: Picks the top K highest scoring experts (e.g. K=2 or K=8).
3. Softmax Normalization: Normalizes the top K gating scores so sum(w_i) == 1.0.
4. Shared Expert Isolation (DeepSeek-V3): Always routes to 1 dedicated shared expert
   for universal domain knowledge, while routing remaining capacity to specialized experts.
"""
import math
from typing import Dict, List, Optional, Tuple
import dataclasses


def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_l = max(logits) if logits else 0.0
    exps = [math.exp(x - max_l) for x in logits]
    sum_exps = sum(exps) or 1.0
    return [e / sum_exps for e in exps]


@dataclasses.dataclass
class RoutedExpertDispatch:
    token_index: int
    selected_expert_ids: List[int]
    expert_weights: List[float]
    shared_expert_active: bool


class MoETopKGatingRouter:
    """
    Simulates Top-K Gating Router for Sparse Mixture of Experts.
    """
    def __init__(self, num_experts: int = 8, top_k: int = 2, has_shared_expert: bool = True):
        self.num_experts = num_experts
        self.top_k = top_k
        self.has_shared_expert = has_shared_expert
        # Simulated routing weight vectors for 8 experts across 4 hidden dimensions
        self.expert_specializations = {
            0: "Python/Code",
            1: "Mathematics",
            2: "Creative Writing",
            3: "System Design/Infra",
            4: "Natural Language QA",
            5: "Logic/Reasoning",
            6: "Multilingual/Translation",
            7: "Science/Physics"
        }

    def route_token(self, token_logits: List[float], token_idx: int = 0) -> RoutedExpertDispatch:
        """
        Routes a single token across experts based on router logits.
        """
        if len(token_logits) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} logits, got {len(token_logits)}")

        # Pair each expert index with its logit score
        indexed_logits = list(enumerate(token_logits))
        
        # Sort by logit descending
        indexed_logits.sort(key=lambda x: x[1], reverse=True)
        
        # Select top K
        top_k_pairs = indexed_logits[:self.top_k]
        top_k_indices = [idx for idx, _ in top_k_pairs]
        top_k_scores = [score for _, score in top_k_pairs]

        # Softmax normalize top K weights
        normalized_weights = softmax(top_k_scores)

        return RoutedExpertDispatch(
            token_index=token_idx,
            selected_expert_ids=top_k_indices,
            expert_weights=normalized_weights,
            shared_expert_active=self.has_shared_expert
        )

    def execute_mock_forward(self, token: str, dispatch: RoutedExpertDispatch) -> str:
        """
        Simulates combining outputs from selected experts.
        """
        expert_contribs = []
        for exp_id, weight in zip(dispatch.selected_expert_ids, dispatch.expert_weights):
            spec = self.expert_specializations.get(exp_id, f"Expert {exp_id}")
            expert_contribs.append(f"[{spec} (w={weight:.2f})]")

        shared_str = " + [Shared Universal Expert (w=1.00)]" if dispatch.shared_expert_active else ""
        return f"Token '{token}' -> Processed by: {' + '.join(expert_contribs)}{shared_str}"


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🔀 SPARSE MIXTURE OF EXPERTS (MoE) TOP-K ROUTER ===\n")

    router = MoETopKGatingRouter(num_experts=8, top_k=2, has_shared_expert=True)

    # Simulated router logits for different input tokens
    test_tokens = [
        ("def", [4.5, 0.2, -1.0, 3.8, 0.1, 0.5, -0.5, 0.0]),         # Code + System Design
        ("integral", [-0.5, 5.2, -2.0, 0.1, 0.3, 4.1, -1.0, 1.2]),    # Math + Logic
        ("Bonjour", [-1.0, -1.0, 1.2, -0.5, 2.0, 0.5, 6.1, 0.0])       # Translation + QA
    ]

    print("Routing tokens across 8 Specialized Experts (Top-2 Active per token):\n")
    for i, (tok, logits) in enumerate(test_tokens):
        dispatch = router.route_token(logits, token_idx=i)
        print(f"Token: {tok!r}")
        print(f"  Selected Expert IDs: {dispatch.selected_expert_ids}")
        print(f"  Normalized Softmax Weights: {[round(w, 3) for w in dispatch.expert_weights]}")
        print(f"  Execution Pipeline: {router.execute_mock_forward(tok, dispatch)}\n")

    print("Key Takeaway: MoE gives 8x parameter capacity with only 2x compute cost per token!")
