# 40. Diffusion Transformers (DiT)

Diffusion Transformers (DiT) replace convolutional U-Nets with Vision Transformers for generative image and video synthesis, powering state-of-the-art systems like OpenAI Sora and FLUX.1.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-dit-patchification` | DiT Patchification & AdaLN-Zero | 2D spatial grid to token sequence, timestep conditioning |
| `02-classifier-free-guidance` | Classifier-Free Guidance (CFG) | Vector extrapolation, prompt alignment vs diversity, reverse ODE |

## Key Concepts

- **Scaling Compute in Generative Media**: Just as LLM loss drops with transformer parameter scaling, generative image quality directly scales with DiT FLOPs.
- **Classifier-Free Guidance (CFG)**: Dual-stream conditional and unconditional evaluation with scale $s$ guarantees sharp adherence to complex prompt prompts.
