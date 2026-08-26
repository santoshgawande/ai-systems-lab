# 38. State Space Models & Mamba

Mamba introduces selective state space models (SSMs) that achieve the modeling power of Transformers while running in linear $O(N)$ training time and constant $O(1)$ inference memory without KV-caches.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-selective-scan-ssm` | Selective State Space Discretization | Continuous SSM math, $\Delta(x_t)$ selection, Zero-Order Hold (ZOH) |
| `02-linear-time-inference` | Constant $O(1)$ Autoregressive Inference | Memory scaling vs Transformers, 0 KV-cache overhead |

## Key Concepts

- **Input-Dependent Selectivity**: Traditional SSMs had time-invariant matrices ($A, B$). Mamba makes parameters dynamic functions of the input token, allowing it to remember or forget selectively.
- **Hardware-Aware Parallel Scan**: Uses a GPU-friendly parallel prefix scan during training to process sequences in parallel despite recurrent semantics.
- **Infinite Generation Length**: Because memory is fixed to state dimension $N$, generation never runs out of memory on long documents.
