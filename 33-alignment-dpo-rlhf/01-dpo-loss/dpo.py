from __future__ import annotations
"""
Direct Preference Optimization (DPO) Loss Engine (Rafailov et al., NeurIPS 2023).

Traditional RLHF requires training an explicit Reward Model (RM) followed by unstable
PPO policy gradient reinforcement learning with high memory overhead.

DPO analytically solves the RLHF objective directly over the policy model parameters theta:
  L_DPO(theta; pi_ref) = - E_{(x, y_w, y_l)} [ log sigma( beta * log(pi_theta(y_w|x)/pi_ref(y_w|x))
                                                        - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ]

Key concepts:
  - beta: Temperature hyperparameter controlling deviation from reference model pi_ref (e.g. 0.1).
  - Implicit reward: r_hat(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x)).
  - Implicit reward margin: r_hat(x, y_w) - r_hat(x, y_l).
"""
import math
from typing import Dict, List, Tuple, Optional
import dataclasses


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x))."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    else:
        return x - math.log1p(math.exp(x))


@dataclasses.dataclass
class DPOTrainSample:
    prompt: str
    chosen: str   # Winning completion y_w
    rejected: str # Losing completion y_l
    pi_theta_chosen_logprob: float
    pi_theta_rejected_logprob: float
    pi_ref_chosen_logprob: float
    pi_ref_rejected_logprob: float


class DPOEngine:
    """
    Computes DPO loss, implicit rewards, and policy drift metrics.
    """
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_implicit_reward(self, policy_logprob: float, ref_logprob: float) -> float:
        """
        Implicit reward: r_hat(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))
        """
        return self.beta * (policy_logprob - ref_logprob)

    def compute_sample_loss(self, sample: DPOTrainSample) -> Tuple[float, float, float]:
        """
        Computes DPO loss for a single pairwise preference sample.
        Returns:
            (dpo_loss, chosen_reward, rejected_reward)
        """
        r_chosen = self.compute_implicit_reward(
            sample.pi_theta_chosen_logprob,
            sample.pi_ref_chosen_logprob
        )
        r_rejected = self.compute_implicit_reward(
            sample.pi_theta_rejected_logprob,
            sample.pi_ref_rejected_logprob
        )
        
        # Reward margin: r_hat(y_w) - r_hat(y_l)
        reward_margin = r_chosen - r_rejected
        
        # Loss = -log(sigmoid(margin))
        loss = -log_sigmoid(reward_margin)
        return loss, r_chosen, r_rejected

    def compute_batch_loss(self, batch: List[DPOTrainSample]) -> Dict[str, float]:
        if not batch:
            return {"loss": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0, "reward_margin": 0.0, "accuracy": 0.0}

        total_loss = 0.0
        total_r_chosen = 0.0
        total_r_rejected = 0.0
        num_correct = 0

        for sample in batch:
            loss, r_w, r_l = self.compute_sample_loss(sample)
            total_loss += loss
            total_r_chosen += r_w
            total_r_rejected += r_l
            if r_w > r_l:
                num_correct += 1

        n = len(batch)
        avg_r_chosen = total_r_chosen / n
        avg_r_rejected = total_r_rejected / n

        return {
            "loss": total_loss / n,
            "avg_chosen_reward": avg_r_chosen,
            "avg_rejected_reward": avg_r_rejected,
            "avg_reward_margin": avg_r_chosen - avg_r_rejected,
            "preference_accuracy": num_correct / n
        }


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ⚖️ DIRECT PREFERENCE OPTIMIZATION (DPO) ENGINE ===\n")

    dpo = DPOEngine(beta=0.1)

    # Simulated training samples with log-probabilities from policy and reference models
    samples = [
        DPOTrainSample(
            prompt="Write a concise function to calculate Fibonacci",
            chosen="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            rejected="Fibonacci is a famous mathematical sequence named after Leonardo of Pisa...",
            pi_theta_chosen_logprob=-5.2,
            pi_ref_chosen_logprob=-6.8,    # Policy prefers chosen more than ref model (+1.6)
            pi_theta_rejected_logprob=-8.5,
            pi_ref_rejected_logprob=-7.1   # Policy suppresses rejected more than ref model (-1.4)
        ),
        DPOTrainSample(
            prompt="How to securely store passwords in database?",
            chosen="Use Argon2id or bcrypt with salt and high work factor.",
            rejected="Store MD5 hashes in the database table.",
            pi_theta_chosen_logprob=-4.1,
            pi_ref_chosen_logprob=-5.5,    # +1.4
            pi_theta_rejected_logprob=-9.2,
            pi_ref_rejected_logprob=-6.0   # -3.2
        )
    ]

    print("Evaluating Pairwise Batch under DPO Loss (beta=0.1):")
    for i, s in enumerate(samples, 1):
        loss, rw, rl = dpo.compute_sample_loss(s)
        print(f"\nSample {i}: Prompt='{s.prompt}'")
        print(f"  Chosen: '{s.chosen}'")
        print(f"  Rejected: '{s.rejected}'")
        print(f"  Implicit Reward Chosen:   r_hat(y_w) = {rw:+.4f}")
        print(f"  Implicit Reward Rejected: r_hat(y_l) = {rl:+.4f}")
        print(f"  Reward Margin (y_w - y_l) : {rw - rl:+.4f}")
        print(f"  Sample DPO Loss: {loss:.4f}")

    metrics = dpo.compute_batch_loss(samples)
    print("\n" + "="*50)
    print(f"Batch Mean Loss:          {metrics['loss']:.4f}")
    print(f"Batch Mean Margin:        {metrics['avg_reward_margin']:.4f}")
    print(f"Batch Alignment Accuracy: {metrics['preference_accuracy']:.1%}")
    print("Takeaway: DPO directly optimizes human preferences with stable binary cross-entropy loss without needing an explicit reward model!")
