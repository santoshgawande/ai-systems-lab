# 33. Alignment & Direct Preference Optimization (DPO)

Alignment bridges the gap between raw pretrained next-token prediction and helpful, harmless, instruction-following AI assistants. Direct Preference Optimization (DPO) has replaced traditional PPO-based RLHF as the gold standard.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-dpo-loss` | DPO Loss & Implicit Rewards | Mathematical derivation, implicit reward calculation, beta temperature |
| `02-preference-dataset` | Preference Dataset Pipeline | Pairwise generation, length-bias mitigation, margin filtering |

## Key Concepts

- **Why Not PPO?**: PPO requires 4 active neural network models in GPU memory (Policy, Reference, Reward Model, Value Critic), leading to high VRAM usage and unstable gradient updates.
- **DPO Analytic Solution**: DPO proves that the optimal policy under the RLHF objective can be parameterized directly via binary cross-entropy on preference pairs.
- **Reference Model Regularization**: Parameter $\beta$ acts as an anchor preventing the model from collapsing into gibberish or diverging too far from the base model $\pi_{ref}$.
