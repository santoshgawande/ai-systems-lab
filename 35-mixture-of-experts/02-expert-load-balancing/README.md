# Lab 02: MoE Expert Load Balancing & Auxiliary Loss

## What You Learn
- The routing collapse problem in Sparse Mixture of Experts.
- Mathematical formulation of the auxiliary load balancing loss: $\mathcal{L}_{aux} = \alpha \cdot E \sum f_i P_i$.
- Enforcing expert capacity factors to prevent GPU memory overflow.
- Detecting token dropping and maintaining uniform GPU parallel utilization.

## Run
```bash
python 02-expert-load-balancing/load_balancer.py
```
