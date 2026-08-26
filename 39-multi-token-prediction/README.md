# 39. Multi-Token Prediction (MTP)

Multi-Token Prediction trains models to predict multiple future tokens simultaneously, improving long-range representation learning and enabling native self-speculative decoding without separate draft models.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-multi-token-heads` | Multi-Token Prediction Architecture | Training $M$ prediction heads over shared trunk, representation quality |
| `02-self-speculative-decoding` | Native Self-Speculative Decoding | Built-in draft proposals, 1-step multi-token acceptance, zero VRAM overhead |

## Key Concepts

- **Mitigating Myopia**: Single next-token prediction causes models to favor local lexical patterns. MTP forces representations to plan global algorithmic structures.
- **Native Speculative Drafting**: MTP heads provide built-in draft candidates during generation, eliminating the need to maintain, load, and synchronize a secondary draft model.
