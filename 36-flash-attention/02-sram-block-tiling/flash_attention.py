from __future__ import annotations
"""
FlashAttention Block Tiling Kernel (Dao et al., 2022 / FlashAttention-2 2023).

Standard Attention:
  1. Load Q, K from HBM -> Compute S = Q K^T / sqrt(d) -> Write S (N x N) to HBM (O(N^2) memory)
  2. Load S from HBM -> Compute P = softmax(S) -> Write P (N x N) to HBM
  3. Load P, V from HBM -> Compute O = P V -> Write O to HBM

FlashAttention:
  Tiling: Divides Q into blocks of size B_r, and K, V into blocks of size B_c.
  Loads blocks directly into fast on-chip SRAM (19 TB/s vs 2 TB/s HBM).
  Computes block attention incrementally using Online Softmax rescaling.
  Zero N x N memory writes to HBM -> Reduces memory footprint from O(N^2) to O(N).
"""
import math
from typing import List, Tuple


def standard_attention(
    Q: List[List[float]],
    K: List[List[float]],
    V: List[List[float]]
) -> List[List[float]]:
    """
    Standard Attention: O = softmax(Q K^T / sqrt(d)) V
    """
    N = len(Q)
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    # 1. S = Q K^T * scale (N x N)
    S = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dot = sum(Q[i][k] * K[j][k] for k in range(d))
            S[i][j] = dot * scale

    # 2. P = softmax(S, dim=-1)
    P = []
    for row in S:
        m = max(row)
        exp_row = [math.exp(val - m) for val in row]
        denom = sum(exp_row)
        P.append([e / denom for e in exp_row])

    # 3. O = P V (N x d)
    O = [[0.0] * d for _ in range(N)]
    for i in range(N):
        for j in range(d):
            O[i][j] = sum(P[i][k] * V[k][j] for k in range(N))

    return O


def flash_attention_tiled(
    Q: List[List[float]],
    K: List[List[float]],
    V: List[List[float]],
    B_r: int = 2,
    B_c: int = 2
) -> List[List[float]]:
    """
    FlashAttention Block-Tiling Kernel using Online Softmax.
    Runs entirely within on-chip SRAM blocks without materializing N x N matrix.
    """
    N = len(Q)
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    # Output accumulator (N x d)
    O = [[0.0] * d for _ in range(N)]
    # Running statistics per query row: max m_i and normalizer l_i
    m = [float("-inf")] * N
    l = [0.0] * N

    # Outer loop over Key/Value blocks (loaded into SRAM)
    for j_block in range(0, N, B_c):
        K_block = K[j_block:j_block + B_c]
        V_block = V[j_block:j_block + B_c]
        actual_Bc = len(K_block)

        # Inner loop over Query blocks (loaded into SRAM)
        for i_block in range(0, N, B_r):
            Q_block = Q[i_block:i_block + B_r]
            actual_Br = len(Q_block)

            for i in range(actual_Br):
                row_idx = i_block + i
                q_vec = Q_block[i]

                # Compute local block dot products S_ij = q_i . K_j * scale
                S_block = [
                    sum(q_vec[k] * K_block[j][k] for k in range(d)) * scale
                    for j in range(actual_Bc)
                ]

                # Online softmax update
                m_prev = m[row_idx]
                l_prev = l[row_idx]

                m_block = max(S_block)
                m_new = max(m_prev, m_block)

                # Compute exponential terms for current block
                P_block = [math.exp(val - m_new) for val in S_block]
                l_block = sum(P_block)

                # Rescale previous accumulator
                if m_prev == float("-inf"):
                    alpha = 0.0
                    l_new = l_block
                else:
                    alpha = math.exp(m_prev - m_new)
                    l_new = l_prev * alpha + l_block

                # Update output accumulator: O_new = (O_prev * alpha + P_block * V_block)
                for dim in range(d):
                    pv_sum = sum(P_block[j] * V_block[j][dim] for j in range(actual_Bc))
                    if m_prev == float("-inf"):
                        O[row_idx][dim] = pv_sum
                    else:
                        O[row_idx][dim] = O[row_idx][dim] * alpha + pv_sum

                # Store updated running stats
                m[row_idx] = m_new
                l[row_idx] = l_new

    # Final normalization by 1 / l_i
    for i in range(N):
        for dim in range(d):
            O[i][dim] /= l[i]

    return O


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🚀 FLASHATTENTION SRAM BLOCK TILING KERNEL ===\n")

    # Sequence length N=4, hidden dimension d=3
    Q = [[1.0, 0.5, 0.2], [0.1, 0.9, 0.4], [0.8, 0.2, 0.7], [0.3, 0.6, 0.5]]
    K = [[0.9, 0.4, 0.1], [0.2, 0.8, 0.3], [0.7, 0.1, 0.6], [0.4, 0.5, 0.5]]
    V = [[10.0, 1.0, 5.0], [2.0, 8.0, 3.0], [7.0, 3.0, 9.0], [4.0, 6.0, 2.0]]

    print(f"Sequence Length N={len(Q)}, Head Dim d={len(Q[0])}")

    # Standard Attention
    O_std = standard_attention(Q, K, V)
    print("\n1. Standard Attention Output (requires O(N^2) HBM memory):")
    for row in O_std:
        print(f"   {[round(v, 4) for v in row]}")

    # FlashAttention Tiled
    O_flash = flash_attention_tiled(Q, K, V, B_r=2, B_c=2)
    print("\n2. FlashAttention-2 Tiled Output (O(N) memory in SRAM):")
    for row in O_flash:
        print(f"   {[round(v, 4) for v in row]}")

    max_diff = max(
        abs(O_std[i][j] - O_flash[i][j])
        for i in range(len(Q))
        for j in range(len(Q[0]))
    )
    print(f"\nMax Absolute Difference: {max_diff:.2e}")
    print("Takeaway: FlashAttention achieves exact mathematical equality to standard attention while eliminating memory I/O bottlenecks!")
