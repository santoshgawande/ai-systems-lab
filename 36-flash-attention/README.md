# 36. FlashAttention & Memory Tiling

FlashAttention is the ubiquitous algorithm powering transformer training and inference across all modern LLMs, eliminating the $O(N^2)$ memory footprint by computing exact attention in fast GPU SRAM.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-online-softmax` | Online Softmax Algorithm | 1-pass streaming softmax, running max and sum updates |
| `02-sram-block-tiling` | FlashAttention-2 Block Tiling | SRAM block loading, online output accumulation, IO-aware scaling |

## Key Concepts

- **IO-Aware Complexity**: Counting memory reads/writes between GPU HBM and SRAM matters more than pure FLOP count.
- **Online Softmax**: Rescaling the partial output matrix dynamically as new key-value blocks are loaded.
- **Zero Approximation**: FlashAttention produces numerically exact attention (not an approximation or sparse heuristic).
