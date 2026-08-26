from __future__ import annotations
"""
Mamba Constant-Memory O(1) Autoregressive Generation.

Transformer generation scaling:
- Token 1:   Reads 1 KV token
- Token 100: Reads 100 KV tokens
- Token 8000: Reads 8,000 KV tokens (Memory footprint grows linearly with sequence length O(N)).

Mamba SSM generation scaling:
- Token 1:   Updates fixed state h_t (size N_state)
- Token 100: Updates fixed state h_t (size N_state)
- Token 8000: Updates fixed state h_t (size N_state) (Memory footprint is STRICTLY CONSTANT O(1)).
"""
import time
from typing import Dict, List, Tuple


class MambaInferenceEngine:
    """
    Simulates memory consumption and step time comparison between Transformers and Mamba.
    """
    def __init__(self, state_dim: int = 16, transformer_kv_dim: int = 128):
        self.state_dim = state_dim
        self.transformer_kv_dim = transformer_kv_dim

    def simulate_generation(self, num_tokens: int = 100) -> Dict[str, List[float]]:
        """
        Simulates generation of N tokens and tracks memory usage per step.
        """
        transformer_memory_floats = []
        mamba_memory_floats = []

        # Current Mamba recurrent state is fixed size
        mamba_state = [0.0] * self.state_dim

        # Transformer KV-cache grows by transformer_kv_dim on every single token
        transformer_cache_size = 0

        for step in range(1, num_tokens + 1):
            # Transformer memory
            transformer_cache_size += self.transformer_kv_dim
            transformer_memory_floats.append(transformer_cache_size)

            # Mamba memory (remains strictly fixed to state_dim)
            mamba_memory_floats.append(self.state_dim)

        return {
            "transformer_memory_per_step": transformer_memory_floats,
            "mamba_memory_per_step": mamba_memory_floats
        }


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ⏱️ MAMBA O(1) CONSTANT-MEMORY INFERENCE BENCHMARK ===\n")

    engine = MambaInferenceEngine(state_dim=64, transformer_kv_dim=256)
    steps = 10

    metrics = engine.simulate_generation(num_tokens=steps)

    print(f"{'Step':<6} | {'Transformer KV Memory (floats)':<32} | {'Mamba SSM Memory (floats)':<28}")
    print("-" * 72)
    for i in range(steps):
        t_mem = metrics["transformer_memory_per_step"][i]
        m_mem = metrics["mamba_memory_per_step"][i]
        print(f"Token {i+1:<2} | {t_mem:<32} | {m_mem:<28}")

    print("\nSummary at Step 10:")
    print(f"  Transformer Memory: {metrics['transformer_memory_per_step'][-1]} floats (Linear O(N) Growth)")
    print(f"  Mamba SSM Memory:   {metrics['mamba_memory_per_step'][-1]} floats (Strictly Constant O(1))")
    print("Takeaway: Mamba generates infinite sequence lengths without running out of GPU memory!")
