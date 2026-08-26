from __future__ import annotations
"""
Classifier-Free Guidance (CFG) Sampling Engine (Ho & Salimans, NeurIPS 2022).

Diffusion models trained solely on text conditioning frequently ignore detailed prompt instructions.
Classifier-Free Guidance (CFG) amplifies prompt adherence without requiring a separate classifier model.

Mathematical Formula:
  e_tilde = e_uncond + scale * (e_cond - e_uncond)
where:
  - e_cond: model prediction conditioned on user prompt c
  - e_uncond: model prediction conditioned on empty null prompt ""
  - scale s: guidance weight (e.g. s = 7.5).
      s = 1.0 -> standard conditional generation
      s > 1.0 -> strongly amplifies prompt semantics and pushes away from generic outputs
"""
from typing import Dict, List, Tuple
import dataclasses


@dataclasses.dataclass
class CFGStepResult:
    timestep: float
    latent_vector: List[float]
    cond_pred: List[float]
    uncond_pred: List[float]
    guided_pred: List[float]


class ClassifierFreeGuidanceEngine:
    """
    Simulates reverse diffusion sampling with Classifier-Free Guidance.
    """
    def __init__(self, guidance_scale: float = 7.5):
        self.guidance_scale = guidance_scale

    def compute_cfg_vector(
        self,
        cond_prediction: List[float],
        uncond_prediction: List[float]
    ) -> List[float]:
        """
        Applies CFG formula: v_guided = v_uncond + scale * (v_cond - v_uncond)
        """
        s = self.guidance_scale
        return [
            uncond + s * (cond - uncond)
            for cond, uncond in zip(cond_prediction, uncond_prediction)
        ]

    def euler_step(
        self,
        latent_x: List[float],
        velocity: List[float],
        dt: float
    ) -> List[float]:
        """
        Integrates reverse ODE trajectory: x_{t-1} = x_t - dt * velocity
        """
        return [x - dt * v for x, v in zip(latent_x, velocity)]


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🌟 CLASSIFIER-FREE GUIDANCE (CFG) SAMPLING ENGINE ===\n")

    cfg_engine = ClassifierFreeGuidanceEngine(guidance_scale=7.5)

    # Simulated noise predictions at timestep t=0.8
    # Prompt: "A hyperrealistic robotic falcon soaring over mountains"
    cond_prediction = [1.2, 0.8, -0.5, 2.0]   # Conditioned on text
    uncond_prediction = [0.2, 0.1, -0.1, 0.4] # Null prompt / unconditional

    print(f"Guidance Scale s={cfg_engine.guidance_scale}")
    print(f"  Unconditioned Noise e_uncond: {uncond_prediction}")
    print(f"  Conditioned Noise   e_cond:   {cond_prediction}")

    guided = cfg_engine.compute_cfg_vector(cond_prediction, uncond_prediction)
    print(f"\nGuided Extrapolated Vector:     {[round(v, 3) for v in guided]}")

    # Simulate 1 Euler reverse diffusion step
    initial_latent = [10.0, 8.0, 5.0, 12.0]
    next_latent = cfg_engine.euler_step(initial_latent, guided, dt=0.1)
    print(f"Latent State after Euler step:  {[round(v, 3) for v in next_latent]}")

    print("\nTakeaway: CFG amplifies semantic features from the prompt, transforming blurry averages into sharp, prompt-faithful images!")
