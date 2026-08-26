# Lab 01: Diffusion Transformer (DiT) Patchification

## What You Learn
- Why modern visual generation models (OpenAI Sora, FLUX.1) replaced U-Nets with Diffusion Transformers.
- Converting 2D spatial latent images ($H \times W \times C$) into 1D token sequences.
- Conditioning self-attention layers on diffusion timesteps using AdaLN-Zero.

## Run
```bash
python 01-dit-patchification/dit_transformer.py
```
