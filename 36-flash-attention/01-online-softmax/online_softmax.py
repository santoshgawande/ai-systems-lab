from __future__ import annotations
"""
Online Softmax Algorithm (Milakov & Gimelshein 2018 / Dao et al. 2022).

Standard Softmax requires 3 separate sequential passes over the entire sequence:
  Pass 1: Find global maximum m = max(x_1, ..., x_N)
  Pass 2: Compute exponential sum d = sum(exp(x_i - m))
  Pass 3: Normalize each element y_i = exp(x_i - m) / d

Online Softmax computes the exact same result in a SINGLE streaming pass by updating
the running maximum and rescaling the running sum as new elements/blocks arrive:
  m_i = max(m_{i-1}, x_i)
  d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
"""
import math
from typing import List, Tuple


def standard_softmax(x: List[float]) -> List[float]:
    """Standard 3-pass softmax for comparison."""
    if not x:
        return []
    m = max(x)
    exp_x = [math.exp(val - m) for val in x]
    d = sum(exp_x)
    return [e / d for e in exp_x]


def online_softmax(x: List[float]) -> Tuple[List[float], float, float]:
    """
    Online 1-pass softmax streaming elements one by one.
    Returns:
        (softmax_probabilities, final_max, final_sum)
    """
    if not x:
        return [], float("-inf"), 0.0

    m = float("-inf")
    d = 0.0

    # Stream elements and update running stats
    for val in x:
        m_prev = m
        m = max(m_prev, val)
        if m_prev == float("-inf"):
            d = math.exp(val - m)
        else:
            d = d * math.exp(m_prev - m) + math.exp(val - m)

    # Normalize elements using final running stats (single pass scaling)
    probs = [math.exp(val - m) / d for val in x]
    return probs, m, d


def chunked_online_softmax(chunks: List[List[float]]) -> Tuple[List[float], float, float]:
    """
    Online softmax across discrete chunks (how FlashAttention processes blocks).
    """
    if not chunks or not any(chunks):
        return [], float("-inf"), 0.0

    m = float("-inf")
    d = 0.0

    for chunk in chunks:
        if not chunk:
            continue
        m_chunk = max(chunk)
        d_chunk = sum(math.exp(val - m_chunk) for val in chunk)

        m_prev = m
        m = max(m_prev, m_chunk)
        if m_prev == float("-inf"):
            d = d_chunk * math.exp(m_chunk - m)
        else:
            d = d * math.exp(m_prev - m) + d_chunk * math.exp(m_chunk - m)

    all_vals = [val for c in chunks for val in c]
    probs = [math.exp(val - m) / d for val in all_vals]
    return probs, m, d


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ⚡ ONLINE SOFTMAX ALGORITHM (FlashAttention Foundation) ===\n")

    input_vector = [2.0, 4.0, 1.0, 8.0, 3.0, 7.0, 5.0, 9.0]
    print(f"Input Logits: {input_vector}\n")

    std_probs = standard_softmax(input_vector)
    print("1. Standard 3-Pass Softmax:")
    print(f"   Probabilities: {[round(p, 4) for p in std_probs]}")
    print(f"   Sum of Probs:  {sum(std_probs):.4f}")

    on_probs, m, d = online_softmax(input_vector)
    print("\n2. Online 1-Pass Streaming Softmax:")
    print(f"   Running Max m: {m:.2f}, Running Normalizer d: {d:.4f}")
    print(f"   Probabilities: {[round(p, 4) for p in on_probs]}")

    # Chunked evaluation (simulating SRAM block loads)
    chunks = [[2.0, 4.0, 1.0, 8.0], [3.0, 7.0, 5.0, 9.0]]
    chunk_probs, _, _ = chunked_online_softmax(chunks)
    print("\n3. Chunked Block-wise Online Softmax:")
    print(f"   Probabilities: {[round(p, 4) for p in chunk_probs]}")

    diff = max(abs(a - b) for a, b in zip(std_probs, on_probs))
    print(f"\nMax Absolute Difference between Standard and Online: {diff:.2e}")
    print("Takeaway: Online Softmax produces mathematically exact softmax without materializing full arrays in memory!")
