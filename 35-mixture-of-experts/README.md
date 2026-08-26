# 35. Sparse Mixture of Experts (MoE)

Sparse Mixture of Experts (MoE) is the foundational architecture behind modern state-of-the-art frontier models like Mixtral 8x7B and DeepSeek-V3, decoupling model parameter count from inference compute cost.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-topk-gating` | Sparse MoE Top-K Gating Router | Top-K expert selection, softmax normalization, shared expert isolation |
| `02-expert-load-balancing` | Expert Load Balancing & Auxiliary Loss | Routing collapse mitigation, $\mathcal{L}_{aux}$ loss calculation, capacity limits |

## Key Concepts

- **Active vs Total Parameters**: A model with 671B parameters only executes 37B active parameters per forward pass, running at the speed and cost of a much smaller model.
- **Auxiliary Loss Regularization**: Without an explicit load balancing loss, routers collapse onto a small subset of experts.
- **DeepSeek Isolated Shared Expert**: Reserving non-routed capacity for foundational cross-domain tasks prevents redundant knowledge duplication across specialized experts.
