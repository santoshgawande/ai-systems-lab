from __future__ import annotations
"""
MoE Expert Load Balancing & Auxiliary Loss Engine (Fedus et al., Switch Transformer / Mixtral).

The Core Challenge in MoE: Routing Collapse.
Left unconstrained, the router quickly learns to send 90%+ of all tokens to 1 or 2 "favorite"
experts. The remaining experts starve, wasting capacity, while the favorite experts bottleneck
GPU execution and cause out-of-memory (OOM) failures.

Solutions:
1. Auxiliary Load Balancing Loss:
     L_aux = alpha * E * sum_{i=1}^E (f_i * P_i)
   where:
     - E: number of experts
     - f_i: fraction of tokens routed to expert i (f_i = count_i / total_tokens)
     - P_i: average routing probability assigned to expert i across the batch
2. Expert Capacity Limits:
     Capacity C = capacity_factor * (num_tokens / E)
   Tokens exceeding capacity are dropped or handled by residual fallback.
"""
from typing import Dict, List, Tuple, Optional
import dataclasses


@dataclasses.dataclass
class LoadBalancingMetrics:
    auxiliary_loss: float
    expert_token_counts: Dict[int, int]
    expert_mean_probs: Dict[int, float]
    dropped_token_count: int
    is_balanced: bool


class MoELoadBalancer:
    """
    Computes auxiliary loss and enforces expert capacity limits.
    """
    def __init__(self, num_experts: int = 8, alpha: float = 0.01, capacity_factor: float = 1.25):
        self.num_experts = num_experts
        self.alpha = alpha  # Auxiliary loss weight
        self.capacity_factor = capacity_factor

    def compute_auxiliary_loss(
        self,
        batch_routing_probs: List[List[float]],  # Shape: [num_tokens, num_experts]
        top_k_indices: List[List[int]]           # Shape: [num_tokens, K]
    ) -> LoadBalancingMetrics:
        """
        Computes L_aux = alpha * E * sum(f_i * P_i)
        """
        num_tokens = len(batch_routing_probs)
        if num_tokens == 0:
            return LoadBalancingMetrics(0.0, {}, {}, 0, True)

        # 1. Calculate expert token counts f_i
        expert_counts = {i: 0 for i in range(self.num_experts)}
        for indices in top_k_indices:
            for exp_id in indices:
                expert_counts[exp_id] += 1

        total_routed_slots = sum(expert_counts.values())
        f_fractions = {i: expert_counts[i] / total_routed_slots for i in range(self.num_experts)}

        # 2. Calculate average router probability P_i per expert across all tokens
        P_probs = {i: 0.0 for i in range(self.num_experts)}
        for token_probs in batch_routing_probs:
            for i, prob in enumerate(token_probs):
                P_probs[i] += prob
        P_fractions = {i: P_probs[i] / num_tokens for i in range(self.num_experts)}

        # 3. Auxiliary loss: alpha * E * sum(f_i * P_i)
        inner_sum = sum(f_fractions[i] * P_fractions[i] for i in range(self.num_experts))
        aux_loss = self.alpha * self.num_experts * inner_sum

        # 4. Check expert capacity limits
        expert_capacity = int(self.capacity_factor * (total_routed_slots / self.num_experts))
        dropped_tokens = 0
        for count in expert_counts.values():
            if count > expert_capacity:
                dropped_tokens += (count - expert_capacity)

        # Ideal uniform fraction is 1 / E
        ideal_fraction = 1.0 / self.num_experts
        is_balanced = all(abs(f_fractions[i] - ideal_fraction) < 0.15 for i in range(self.num_experts))

        return LoadBalancingMetrics(
            auxiliary_loss=aux_loss,
            expert_token_counts=expert_counts,
            expert_mean_probs=P_fractions,
            dropped_token_count=dropped_tokens,
            is_balanced=is_balanced
        )


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ⚖️ MoE EXPERT LOAD BALANCER & AUXILIARY LOSS ===\n")

    balancer = MoELoadBalancer(num_experts=4, alpha=0.01, capacity_factor=1.2)

    print("Scenario 1: Severely Collapsed Routing (All tokens routed to Expert 0):")
    collapsed_probs = [[0.9, 0.05, 0.03, 0.02] for _ in range(10)]
    collapsed_indices = [[0] for _ in range(10)]
    m1 = balancer.compute_auxiliary_loss(collapsed_probs, collapsed_indices)
    print(f"  Expert Token Counts: {m1.expert_token_counts}")
    print(f"  Auxiliary Loss:      {m1.auxiliary_loss:.6f} (High Penalty)")
    print(f"  Dropped Tokens:      {m1.dropped_token_count}")
    print(f"  Is Balanced:         {m1.is_balanced}")

    print("\nScenario 2: Perfectly Balanced Routing (Uniform distribution across all 4 experts):")
    balanced_probs = [[0.25, 0.25, 0.25, 0.25] for _ in range(12)]
    balanced_indices = [[i % 4] for i in range(12)]
    m2 = balancer.compute_auxiliary_loss(balanced_probs, balanced_indices)
    print(f"  Expert Token Counts: {m2.expert_token_counts}")
    print(f"  Auxiliary Loss:      {m2.auxiliary_loss:.6f} (Low Minimal Penalty)")
    print(f"  Dropped Tokens:      {m2.dropped_token_count}")
    print(f"  Is Balanced:         {m2.is_balanced}")

    print("\nKey Takeaway: Auxiliary load balancing loss penalizes router concentration and forces uniform GPU workload distribution!")
