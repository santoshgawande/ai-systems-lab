from __future__ import annotations
"""
Megatron-LM Tensor Parallelism (Shoeybi et al., NVIDIA 2019).

Training and serving 70B+ parameter models exceeds the VRAM of a single GPU.
Tensor Parallelism (TP) splits individual weight matrices across GPUs (e.g., TP=2 or TP=8)
such that communication operations (All-Reduce) are minimized.

Megatron-LM MLP Layer Formulation:
1. Column-Parallel Linear Layer:
     Splits weight matrix W_1 along columns: W_1 = [W_{1,1} | W_{1,2}]
     GPU 0 computes: Y_0 = GeLU(X . W_{1,1})
     GPU 1 computes: Y_1 = GeLU(X . W_{1,2})
     (ZERO communication required during intermediate activation!)
2. Row-Parallel Linear Layer:
     Splits weight matrix W_2 along rows: W_2 = [W_{2,1} / W_{2,2}]
     GPU 0 computes: Z_0 = Y_0 . W_{2,1}
     GPU 1 computes: Z_1 = Y_1 . W_{2,2}
3. All-Reduce Sum:
     Sums outputs across GPUs: Output = Z_0 + Z_1 (Single communication barrier per MLP block!)
"""
from typing import List, Tuple
import math


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def single_gpu_dense_mlp(
    X: List[float],             # Input vector (dim d_in)
    W1: List[List[float]],      # Shape: [d_in, d_hidden]
    W2: List[List[float]]       # Shape: [d_hidden, d_out]
) -> List[float]:
    """
    Standard Single-GPU dense MLP computation.
    """
    d_in = len(X)
    d_hidden = len(W1[0])
    d_out = len(W2[0])

    # 1. Y = GeLU(X * W1)
    Y = [gelu(sum(X[i] * W1[i][j] for i in range(d_in))) for j in range(d_hidden)]

    # 2. Z = Y * W2
    Z = [sum(Y[j] * W2[j][k] for j in range(d_hidden)) for k in range(d_out)]
    return Z


def megatron_tensor_parallel_mlp(
    X: List[float],             # Input vector (dim d_in)
    W1: List[List[float]],      # Shape: [d_in, d_hidden]
    W2: List[List[float]],      # Shape: [d_hidden, d_out]
    tp_world_size: int = 2
) -> Tuple[List[float], List[List[float]]]:
    """
    Simulates TP=2 Column-Row Parallel MLP execution.
    Returns:
        (all_reduce_output, [gpu0_partial_out, gpu1_partial_out])
    """
    d_in = len(X)
    d_hidden = len(W1[0])
    d_out = len(W2[0])
    split_h = d_hidden // tp_world_size

    gpu_partials = []

    # Simulate execution on each TP rank in parallel
    for rank in range(tp_world_size):
        h_start = rank * split_h
        h_end = (rank + 1) * split_h

        # 1. Column Parallel W1 slice: [d_in, split_h]
        W1_rank = [[W1[i][j] for j in range(h_start, h_end)] for i in range(d_in)]
        Y_rank = [gelu(sum(X[i] * W1_rank[i][j] for i in range(d_in))) for j in range(split_h)]

        # 2. Row Parallel W2 slice: [split_h, d_out]
        W2_rank = [[W2[j][k] for k in range(d_out)] for j in range(h_start, h_end)]
        Z_rank = [sum(Y_rank[j] * W2_rank[j][k] for j in range(split_h)) for k in range(d_out)]

        gpu_partials.append(Z_rank)

    # 3. All-Reduce Sum across TP ranks
    all_reduce_output = [
        sum(gpu_partials[rank][k] for rank in range(tp_world_size))
        for k in range(d_out)
    ]

    return all_reduce_output, gpu_partials


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🌐 MEGATRON-LM TENSOR PARALLELISM (TP=2) ===\n")

    # Dimensions: d_in=2, d_hidden=4, d_out=2
    X = [1.5, -0.5]
    W1 = [
        [0.8, -0.2, 0.4, 0.9],
        [-0.5, 0.6, 0.1, -0.7]
    ]
    W2 = [
        [0.5, -0.4],
        [0.2, 0.8],
        [-0.9, 0.1],
        [0.3, 0.6]
    ]

    print(f"Input Vector X: {X}")

    # Baseline: Single GPU
    out_single = single_gpu_dense_mlp(X, W1, W2)
    print(f"1. Single GPU Dense MLP Output: {[round(v, 6) for v in out_single]}")

    # Tensor Parallel (TP=2)
    out_tp, partials = megatron_tensor_parallel_mlp(X, W1, W2, tp_world_size=2)
    print(f"\n2. Megatron TP=2 Execution:")
    print(f"   GPU 0 (Column W1_0 -> Row W2_0) Partial: {[round(v, 6) for v in partials[0]]}")
    print(f"   GPU 1 (Column W1_1 -> Row W2_1) Partial: {[round(v, 6) for v in partials[1]]}")
    print(f"   All-Reduce Sum Output (GPU0 + GPU1):    {[round(v, 6) for v in out_tp]}")

    diff = max(abs(a - b) for a, b in zip(out_single, out_tp))
    print(f"\nMax Absolute Difference: {diff:.2e}")
    print("Takeaway: Megatron Column-Row parallelism requires only ONE All-Reduce per transformer block!")
