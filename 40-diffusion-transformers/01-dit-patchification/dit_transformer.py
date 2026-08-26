from __future__ import annotations
"""
Diffusion Transformer (DiT) Architecture & Patchification (Peebles & Xie, ICCV 2023).

Prior diffusion models (Stable Diffusion 1.5/XL) used convolutional U-Net backbones.
State-of-the-art visual generation (OpenAI Sora, FLUX.1, Midjourney v6) replaced U-Nets with
Diffusion Transformers (DiT) due to superior compute scaling laws.

Key Primitives:
1. Patchification: Flattens spatial latent grid (H x W x C) into a sequence of (p x p) patches.
2. AdaLN-Zero (Adaptive LayerNorm): Conditions attention blocks on timestep t and text embedding c
   by predicting scale (gamma), shift (beta), and gate (alpha) vectors initialized to zero.
"""
from typing import Dict, List, Tuple
import dataclasses
import math


@dataclasses.dataclass
class DiTPatchSequence:
    patches: List[List[float]]  # Sequence of flattened patches [num_patches, patch_dim]
    num_patches_h: int
    num_patches_w: int
    patch_size: int
    channels: int


class DiffusionTransformerEngine:
    """
    Simulates DiT patchification, timestep embedding, and AdaLN-Zero modulation.
    """
    def __init__(self, patch_size: int = 2, hidden_dim: int = 16):
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def patchify_latent(
        self,
        image_grid: List[List[List[float]]]  # Shape: [H, W, C]
    ) -> DiTPatchSequence:
        """
        Converts 2D spatial grid [H, W, C] into a 1D sequence of patch tokens.
        """
        H = len(image_grid)
        W = len(image_grid[0])
        C = len(image_grid[0][0])
        p = self.patch_size

        if H % p != 0 or W % p != 0:
            raise ValueError(f"Image dimensions ({H}x{W}) must be divisible by patch size {p}")

        num_h = H // p
        num_w = W // p
        patches: List[List[float]] = []

        for i in range(num_h):
            for j in range(num_w):
                # Flatten p x p x C patch into 1D vector
                patch_vec = []
                for dy in range(p):
                    for dx in range(p):
                        patch_vec.extend(image_grid[i * p + dy][j * p + dx])
                patches.append(patch_vec)

        return DiTPatchSequence(
            patches=patches,
            num_patches_h=num_h,
            num_patches_w=num_w,
            patch_size=p,
            channels=C
        )

    def adaln_zero_modulate(
        self,
        patch_token: List[float],
        timestep: float,
        condition_vector: List[float]
    ) -> List[float]:
        """
        Applies AdaLN-Zero modulation: y = alpha * (LayerNorm(x) * (1 + gamma) + beta)
        where gamma, beta, alpha are dynamically conditioned on timestep and prompt.
        """
        # Timestep sinusoidal embedding
        t_embed = math.sin(timestep / 100.0)
        
        # Predicted scale gamma, shift beta, gate alpha
        gamma = [t_embed * 0.1 * c for c in condition_vector[:len(patch_token)]]
        beta = [t_embed * 0.05 * c for c in condition_vector[:len(patch_token)]]
        alpha = [0.9 + 0.1 * t_embed for _ in range(len(patch_token))]

        # Apply modulation
        modulated = [
            alpha[i] * (patch_token[i] * (1.0 + gamma[i]) + beta[i])
            for i in range(len(patch_token))
        ]
        return modulated


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🎨 DIFFUSION TRANSFORMER (DiT) PATCHIFICATION ===\n")

    engine = DiffusionTransformerEngine(patch_size=2, hidden_dim=8)

    # Simulated 4x4 spatial latent image with 2 channels
    mock_latent_grid = [
        [[1.0, 0.5], [1.2, 0.6], [0.1, 0.9], [0.2, 0.8]],
        [[1.1, 0.4], [1.3, 0.7], [0.3, 0.8], [0.4, 0.7]],
        [[0.8, 0.2], [0.9, 0.3], [2.0, 1.5], [2.1, 1.6]],
        [[0.7, 0.1], [0.8, 0.2], [2.2, 1.4], [2.3, 1.7]]
    ]

    print("1. Ingesting 4x4x2 Spatial Latent Grid...")
    patch_seq = engine.patchify_latent(mock_latent_grid)
    print(f"   Generated {len(patch_seq.patches)} Patches of size (2x2x2 = 8 floats each):")
    for idx, p in enumerate(patch_seq.patches, 1):
        print(f"     Patch [{idx:02d}]: {[round(v, 2) for v in p]}")

    print("\n2. Applying AdaLN-Zero Conditioning (Timestep t=50.0):")
    cond = [1.0] * 8
    sample_patch = patch_seq.patches[0]
    mod_patch = engine.adaln_zero_modulate(sample_patch, timestep=50.0, condition_vector=cond)
    print(f"   Original Patch:   {[round(v, 3) for v in sample_patch]}")
    print(f"   Modulated Patch:  {[round(v, 3) for v in mod_patch]}")
    print("\nTakeaway: DiT treats visual latents as standard token sequences, unlocking scalable transformer scaling!")
