from __future__ import annotations
"""
Medusa Multi-Head Speculative Decoding (Cai et al., 2024).

Unlike standard speculative decoding which requires a separate draft model,
Medusa adds multiple lightweight Feed-Forward (MLP) prediction heads on top of the
SAME base model backbone.

Head 1 predicts token at t + 1
Head 2 predicts token at t + 2
Head 3 predicts token at t + 3

Candidates are generated in parallel, evaluated with Tree Attention, and the longest
valid prefix is accepted in a single forward pass.
"""
from typing import Dict, List, Tuple, Optional
import dataclasses


@dataclasses.dataclass
class MedusaCandidate:
    path: List[str]
    score: float
    depth: int


class MedusaEngine:
    """
    Simulates Medusa Multi-Head decoding and tree verification.
    """
    def __init__(self, num_heads: int = 3, topk_per_head: int = 2):
        self.num_heads = num_heads
        self.topk_per_head = topk_per_head
        self.total_accepted_tokens = 0
        self.forward_steps = 0

    def generate_candidate_tree(
        self,
        base_token: str,
        head_predictions: List[List[Tuple[str, float]]]
    ) -> List[MedusaCandidate]:
        """
        Builds candidate paths across Medusa heads.
        Example with 2 heads top-2:
        Root: "import"
        Head 1: ["os", "sys"]
        Head 2: [";", "\n"]
        Paths: ["os", ";"], ["os", "\n"], ["sys", ";"], ["sys", "\n"]
        """
        candidates: List[MedusaCandidate] = []

        def build_paths(current_path: List[str], current_score: float, head_idx: int):
            if head_idx >= len(head_predictions) or head_idx >= self.num_heads:
                if current_path:
                    candidates.append(MedusaCandidate(
                        path=list(current_path),
                        score=current_score,
                        depth=len(current_path)
                    ))
                return

            for tok, prob in head_predictions[head_idx][:self.topk_per_head]:
                build_paths(current_path + [tok], current_score * prob, head_idx + 1)

        build_paths([], 1.0, 0)
        # Sort candidates by combined confidence score
        candidates.sort(key=lambda c: (c.depth, c.score), reverse=True)
        return candidates

    def verify_longest_prefix(
        self,
        ground_truth_seq: List[str],
        candidates: List[MedusaCandidate]
    ) -> Tuple[List[str], int]:
        """
        Verifies which candidate tree branch matches the ground truth backbone logits.
        Returns the longest accepted sequence of tokens.
        """
        best_accepted: List[str] = []

        for candidate in candidates:
            # Check if this candidate matches ground truth up to candidate length
            matching_prefix = []
            for pred_tok, gt_tok in zip(candidate.path, ground_truth_seq):
                if pred_tok == gt_tok:
                    matching_prefix.append(pred_tok)
                else:
                    break
            
            if len(matching_prefix) > len(best_accepted):
                best_accepted = matching_prefix

        # If no head predicted correctly, accept at least 1 ground truth token
        if not best_accepted and ground_truth_seq:
            best_accepted = [ground_truth_seq[0]]

        self.total_accepted_tokens += len(best_accepted)
        self.forward_steps += 1
        return best_accepted, len(best_accepted)


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🐍 MEDUSA MULTI-HEAD SPECULATIVE DECODING ===\n")

    engine = MedusaEngine(num_heads=3, topk_per_head=2)

    # Step 1: Base token is "def"
    # Head 1 (t+1), Head 2 (t+2), Head 3 (t+3)
    head_preds = [
        [("binary_search", 0.8), ("quicksort", 0.2)],
        [("(", 0.9), ("[", 0.1)],
        [("arr", 0.85), ("target", 0.15)]
    ]

    print("Step 1: Generating candidate tree from 3 Medusa heads...")
    candidates = engine.generate_candidate_tree("def", head_preds)
    for i, c in enumerate(candidates, 1):
        print(f"  Branch {i:02d} [depth={c.depth}]: {' -> '.join(c.path)} (conf={c.score:.4f})")

    # True continuation from backbone logits
    ground_truth = ["binary_search", "(", "arr", ",", "target", ")"]
    print(f"\nTarget Ground Truth continuation: {ground_truth}")

    accepted, count = engine.verify_longest_prefix(ground_truth, candidates)
    print(f"Accepted prefix in 1 forward pass: {accepted} (+{count} tokens)")
    print(f"Medusa Speedup efficiency: {count:.2f} tokens/step (vs 1 token/step standard autoregression)\n")
