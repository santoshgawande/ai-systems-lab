# Lab 01: Direct Preference Optimization (DPO) Loss

## What You Learn
- Why DPO replaced PPO and explicit Reward Models in modern LLM post-training.
- The mathematical derivation of the DPO loss objective: $-\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$.
- Computing implicit rewards $\hat{r}_\theta(x, y)$ directly from policy vs reference log-probabilities.
- The role of the KL-divergence penalty hyperparameter $\beta$.

## Run
```bash
python 01-dpo-loss/dpo.py
```
