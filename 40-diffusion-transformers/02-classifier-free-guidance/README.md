# Lab 02: Classifier-Free Guidance (CFG)

## What You Learn
- Why standard conditional diffusion models produce washed out, generic images.
- The mathematical derivation of the CFG vector extrapolation formula:
  $$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$
- Balancing guidance scale $s$ against sample diversity and artifacts.
- Integrating trajectories using reverse Euler ODE steps.

## Run
```bash
python 02-classifier-free-guidance/cfg_sampling.py
```
