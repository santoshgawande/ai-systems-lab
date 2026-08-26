from __future__ import annotations
"""
Selective State Space Model (SSM) Scan Engine (Gu & Dao, NeurIPS 2023 / Mamba).

Transformers suffer from O(N^2) training and inference compute costs due to attention.
Linear RNNs process tokens in O(N) time but lack the ability to selectively filter or focus.

Mamba introduces Input-Dependent Selective Discretization:
  Continuous state-space equations:
    h'(t) = A h(t) + B x(t)
    y(t)  = C h(t)
  Selective discretization (Zero-Order Hold with input-dependent step size Delta):
    Delta_t = softplus(Parameter + Linear_Delta(x_t))
    B_t     = Linear_B(x_t)
    C_t     = Linear_C(x_t)
    A_bar   = exp(Delta_t * A)
    B_bar   = Delta_t * B_t

Recurrent update:
  h_t = A_bar * h_{t-1} + B_bar * x_t
  y_t = C_t * h_t
"""
import math
from typing import List, Tuple
import dataclasses


def softplus(x: float) -> float:
    """softplus(x) = log(1 + exp(x))"""
    if x > 20:
        return x
    return math.log1p(math.exp(x))


@dataclasses.dataclass
class SSMStepResult:
    hidden_state: List[float]  # Recurrent hidden state h_t (dim N_state)
    output: float              # Emitted scalar y_t
    delta_t: float             # Selectivity gate (retention factor)


class SelectiveSSMBlock:
    """
    Selective State Space Model block with state dimension N_state.
    """
    def __init__(self, state_dim: int = 4, dt_rank: float = 0.5):
        self.state_dim = state_dim
        # Base structured transition matrix A (typically diagonal negative real numbers)
        self.A = [-1.0 * (i + 1) for i in range(state_dim)]
        self.dt_bias = 0.1

    def discretize(self, x_t: float) -> Tuple[List[float], List[float], float]:
        """
        Computes input-dependent Delta_t, A_bar, and B_bar.
        """
        # Delta_t is input-dependent: higher x_t -> larger step size -> more new info absorbed
        delta_t = softplus(self.dt_bias + 0.5 * abs(x_t))
        
        # A_bar = exp(Delta_t * A) (retention decay)
        A_bar = [math.exp(delta_t * a_val) for a_val in self.A]
        
        # B_bar = Delta_t * B_t (where B_t is input projection)
        B_t = [0.5 * (i + 1) for i in range(self.state_dim)]
        B_bar = [delta_t * b_val * x_t for b_val in B_t]

        return A_bar, B_bar, delta_t

    def step(self, h_prev: List[float], x_t: float) -> SSMStepResult:
        """
        Single recurrent step: h_t = A_bar * h_{prev} + B_bar * x_t, y_t = C_t * h_t
        """
        A_bar, B_bar, delta_t = self.discretize(x_t)

        # Recurrent state update
        h_t = [
            A_bar[i] * h_prev[i] + B_bar[i]
            for i in range(self.state_dim)
        ]

        # Output projection: y_t = C_t . h_t
        C_t = [1.0 / (i + 1) for i in range(self.state_dim)]
        y_t = sum(c * h for c, h in zip(C_t, h_t))

        return SSMStepResult(hidden_state=h_t, output=y_t, delta_t=delta_t)

    def scan_sequence(self, sequence: List[float]) -> List[SSMStepResult]:
        """
        Processes an entire sequence sequentially in linear O(N) time.
        """
        h_curr = [0.0] * self.state_dim
        results = []
        for token_val in sequence:
            res = self.step(h_curr, token_val)
            results.append(res)
            h_curr = res.hidden_state
        return results


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🐍 MAMBA SELECTIVE STATE SPACE SCAN (Gu & Dao) ===\n")

    ssm = SelectiveSSMBlock(state_dim=4)
    tokens = [1.0, 2.5, 0.1, -1.5, 3.0]

    print(f"Processing sequence of length N={len(tokens)}: {tokens}")
    print(f"SSM State Dimension: {ssm.state_dim} (Fixed memory size throughout generation)\n")

    results = ssm.scan_sequence(tokens)
    for t, (tok, res) in enumerate(zip(tokens, results), 1):
        state_str = [round(h, 4) for h in res.hidden_state]
        print(f"Step {t}: Token={tok:+.1f} -> Delta={res.delta_t:.3f} | Output y_t={res.output:+.4f} | State h_t={state_str}")

    print("\nTakeaway: Mamba compresses sequence history into a fixed-size recurrent state in linear O(N) time with 0 KV-cache overhead!")
