from __future__ import annotations
"""
Multi-Head Latent Attention (MLA) KV-Cache Compression (DeepSeek-V2 / V3 / R1).

The KV-cache memory bottleneck limits context length and batch size:
- Standard Multi-Head Attention (MHA): Caches 2 * n_h * d_h per token (e.g. 16,384 floats).
- Grouped-Query Attention (GQA): Reduces number of KV heads to G (e.g., Llama 3 8B with G=8),
  which reduces memory but hurts capacity on complex reasoning tasks.

DeepSeek's Multi-Head Latent Attention (MLA):
- Compresses Keys and Values jointly into a tiny latent vector c_t^{KV} of dimension d_c (e.g. 512).
- Decouples RoPE rotary position embeddings into a separate tiny vector k_t^R of dimension d_R (64).
- Total KV-cache per token = d_c + d_R = 576 floats (93.3% reduction vs MHA!).
"""
from typing import Dict, List, Tuple
import dataclasses
import math


@dataclasses.dataclass
class MLATokenKVCache:
    latent_kv: List[float]   # Compressed latent vector c_t^{KV} (dim d_c)
    decoupled_rope_key: List[float] # Position-aware key k_t^R (dim d_R)

    @property
    def total_floats(self) -> int:
        return len(self.latent_kv) + len(self.decoupled_rope_key)


class MultiHeadLatentAttentionEngine:
    """
    Simulates Multi-Head Latent Attention low-rank compression and cache management.
    """
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        latent_kv_dim: int = 64,   # Low-rank compression dimension d_c
        rope_dim: int = 16          # Decoupled RoPE dimension d_R
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_kv_dim = latent_kv_dim
        self.rope_dim = rope_dim

        # Standard MHA KV floats per token: 2 * num_heads * head_dim
        self.mha_kv_per_token = 2 * num_heads * head_dim
        # MLA KV floats per token: latent_kv_dim + rope_dim
        self.mla_kv_per_token = latent_kv_dim + rope_dim

    def compress_kv(self, hidden_state: List[float], pos_idx: int) -> MLATokenKVCache:
        """
        Projects hidden state h_t into low-rank latent vector c_t^{KV} and decoupled key k_t^R.
        """
        # Simulated linear projection to latent dimension d_c
        latent = [
            sum(hidden_state[j] * 0.05 for j in range(i, min(len(hidden_state), i + 8)))
            for i in range(self.latent_kv_dim)
        ]
        # Decoupled RoPE position embedding
        rope_key = [
            math.sin(pos_idx / (10000 ** (2 * d / self.rope_dim)))
            for d in range(self.rope_dim)
        ]

        return MLATokenKVCache(latent_kv=latent, decoupled_rope_key=rope_key)

    def compute_memory_savings(self, sequence_length: int, batch_size: int = 32) -> Dict[str, float]:
        """
        Compares VRAM footprint between standard MHA, GQA (G=8), and MLA.
        """
        # In MegaBytes (assuming float16 = 2 bytes per element)
        bytes_per_float = 2
        mha_mb = (self.mha_kv_per_token * sequence_length * batch_size * bytes_per_float) / (1024 * 1024)
        gqa_mb = ((2 * 8 * self.head_dim) * sequence_length * batch_size * bytes_per_float) / (1024 * 1024)
        mla_mb = (self.mla_kv_per_token * sequence_length * batch_size * bytes_per_float) / (1024 * 1024)

        compression_ratio = self.mha_kv_per_token / self.mla_kv_per_token
        savings_pct = (1.0 - (self.mla_kv_per_token / self.mha_kv_per_token)) * 100.0

        return {
            "mha_vram_mb": mha_mb,
            "gqa_vram_mb": gqa_mb,
            "mla_vram_mb": mla_mb,
            "compression_ratio": compression_ratio,
            "savings_percentage": savings_pct
        }


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 📉 DEEPSEEK MULTI-HEAD LATENT ATTENTION (MLA) ===\n")

    engine = MultiHeadLatentAttentionEngine(
        hidden_dim=2048,
        num_heads=16,
        head_dim=128,
        latent_kv_dim=128,
        rope_dim=32
    )

    print(f"Architecture Parameters:")
    print(f"  Hidden Dim: {engine.hidden_dim}, Heads: {engine.num_heads}, Head Dim: {engine.head_dim}")
    print(f"  Standard MHA KV-Cache: {engine.mha_kv_per_token} floats/token")
    print(f"  DeepSeek MLA KV-Cache: {engine.mla_kv_per_token} floats/token (128 latent + 32 RoPE)")

    # Simulate compression of 1 token
    mock_hidden = [0.5] * 2048
    cached_token = engine.compress_kv(mock_hidden, pos_idx=1)
    print(f"\n1. Token KV Compressed Representation:")
    print(f"   Stored Floats in KV Cache: {cached_token.total_floats} floats")

    # VRAM Comparison on 32k context length
    stats = engine.compute_memory_savings(sequence_length=32768, batch_size=16)
    print(f"\n2. Memory Comparison at 32k Context (Batch=16, Float16):")
    print(f"   Standard MHA KV-Cache: {stats['mha_vram_mb']:.1f} MB ({stats['mha_vram_mb']/1024:.2f} GB)")
    print(f"   Grouped-Query (GQA 8): {stats['gqa_vram_mb']:.1f} MB ({stats['gqa_vram_mb']/1024:.2f} GB)")
    print(f"   DeepSeek MLA KV-Cache: {stats['mla_vram_mb']:.1f} MB ({stats['mla_vram_mb']/1024:.2f} GB)")
    print(f"   VRAM Memory Savings:   {stats['savings_percentage']:.1f}% ({stats['compression_ratio']:.1f}x compression)")

    print("\nTakeaway: MLA enables 10x larger batch sizes and long contexts by compressing the KV cache while keeping all 16 attention heads fully expressive!")
