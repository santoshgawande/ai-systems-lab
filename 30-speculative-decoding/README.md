# 30. Speculative Decoding

Speculative decoding is one of the most effective inference acceleration techniques for production LLMs, achieving $2\times$ to $3.5\times$ wall-clock speedups with zero degradation in generation quality.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-speculative-sampling` | Speculative Sampling (Leviathan et al.) | Draft-target rejection sampling, residual sampling, speedup math |
| `02-medusa-heads` | Medusa Multi-Head Decoding | Multiple heads on single base model, tree attention, candidate branches |

## Key Concepts

- **Memory Bandwidth Bottleneck**: Autoregressive decoding is memory-bandwidth bound (reading 70B weights per 1 token). Speculative decoding turns memory-bound generation into compute-bound parallel verification.
- **Strict Equivalence**: Speculative sampling provably guarantees that the output follows the exact probability distribution $q(x)$ of the target model.
- **Acceptance Rate $\alpha$**: Depends on alignment between draft and target model; typical $\alpha \in [0.65, 0.85]$ yielding $2.5\times$ speedup.
