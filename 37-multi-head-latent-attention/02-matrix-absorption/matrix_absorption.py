from __future__ import annotations
"""
Multi-Head Latent Attention (MLA) Matrix Absorption (DeepSeek-V2 / V3).

During inference decoding, reconstructing the full uncompressed Key matrix K (dim n_h * d_h)
from the cached latent vector c^{KV} (dim d_c) on every step would waste compute memory bandwidth.

Mathematical Proof of Matrix Absorption:
  Standard Attention score:
    Score = q_t^T . k_j = q_t^T (W^{UK} . c_j^{KV})
  Associativity of Matrix Multiplication:
    Score = (q_t^T . W^{UK}) . c_j^{KV} = ( (W^{UK})^T . q_t )^T . c_j^{KV} = q'_{t}^T . c_j^{KV}

Key Insight:
  We absorb the Key up-projection matrix W^{UK} directly into the Query projection W^Q!
  We project the single query vector q_t directly into the latent space (dim d_c) ONCE,
  and then take direct dot products against the small cached latent vectors c_j^{KV}.
  Keys are NEVER decompressed into full head dimensions during inference!
"""
from typing import List, Tuple


def standard_unabsorbed_attention_score(
    q: List[float],             # Query vector (dim head_dim d)
    c_kv: List[float],          # Cached latent vector (dim d_c)
    W_UK: List[List[float]]     # Key Up-projection matrix (shape d x d_c)
) -> float:
    """
    Standard path: Decompress latent c_kv -> full key k = W_UK * c_kv -> dot product q . k
    """
    d = len(q)
    d_c = len(c_kv)

    # 1. Decompress key k (dim d)
    k = [sum(W_UK[i][j] * c_kv[j] for j in range(d_c)) for i in range(d)]

    # 2. Dot product q . k
    score = sum(q[i] * k[i] for i in range(d))
    return score


def absorbed_attention_score(
    q: List[float],             # Query vector (dim head_dim d)
    c_kv: List[float],          # Cached latent vector (dim d_c)
    W_UK: List[List[float]]     # Key Up-projection matrix (shape d x d_c)
) -> float:
    """
    Absorbed path: Pre-project query q' = W_UK^T * q (dim d_c) -> dot product q' . c_kv
    """
    d = len(q)
    d_c = len(c_kv)

    # 1. Transform query directly into latent space q' (dim d_c)
    q_absorbed = [sum(W_UK[i][j] * q[i] for i in range(d)) for j in range(d_c)]

    # 2. Direct dot product q' . c_kv in latent space (O(d_c) ops instead of O(d))
    score = sum(q_absorbed[j] * c_kv[j] for j in range(d_c))
    return score


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🧮 MLA INFERENCE MATRIX ABSORPTION PROOF ===\n")

    # Dimensions: head_dim d=6, latent_dim d_c=3
    q = [1.2, -0.5, 0.8, 2.1, -1.0, 0.4]
    c_kv = [0.7, -0.3, 1.5]

    # Matrix W_UK (6 x 3)
    W_UK = [
        [0.1, 0.4, -0.2],
        [0.5, -0.1, 0.3],
        [-0.3, 0.2, 0.8],
        [0.9, 0.0, -0.5],
        [0.2, -0.6, 0.1],
        [-0.4, 0.7, 0.3]
    ]

    print("Comparing Attention Score Computation:")
    print(f"  Query Vector q:       {q} (dim={len(q)})")
    print(f"  Cached Latent c_kv:   {c_kv} (dim={len(c_kv)})\n")

    # 1. Unabsorbed
    score_unabs = standard_unabsorbed_attention_score(q, c_kv, W_UK)
    print(f"1. Unabsorbed Path (Decompressing full Key k): Score = {score_unabs:.6f}")

    # 2. Absorbed
    score_abs = absorbed_attention_score(q, c_kv, W_UK)
    print(f"2. Absorbed Path (Projecting Query directly):  Score = {score_abs:.6f}")

    diff = abs(score_unabs - score_abs)
    print(f"\nExact Mathematical Equality Difference: {diff:.2e}")
    print("Takeaway: Matrix absorption computes exact attention scores directly against cached compressed latents!")
