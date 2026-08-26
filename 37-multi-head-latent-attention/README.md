# 37. Multi-Head Latent Attention (MLA)

Multi-Head Latent Attention (MLA) is DeepSeek's architectural breakthrough that solves the KV-cache bottleneck of modern LLMs, enabling massive 128k+ context windows with 93% smaller memory footprints.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-latent-kv-compression` | Joint KV Low-Rank Compression | Latent vector $c_t^{KV}$, decoupled RoPE keys $k_t^R$, VRAM savings |
| `02-matrix-absorption` | Inference Matrix Absorption | Absorbing $W^{UK}$ into query projection, direct latent dot products |

## Key Concepts

- **Beyond GQA**: GQA reduces memory by dropping key-value heads, but sacrifices model capacity on difficult reasoning. MLA preserves all attention heads while compressing the underlying representations.
- **Decoupled RoPE Key**: RoPE rotary matrices cannot be linearly compressed without losing position sensitivity; decoupling position keys solves this fundamental limitation.
- **Inference Efficiency**: Matrix absorption eliminates runtime key decompression, allowing generation at native matrix speeds.
