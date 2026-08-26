# Lab 01: Mamba Selective State Space Scan

## What You Learn
- Why standard linear SSMs (S4) could not match Transformer performance on language modeling.
- The mathematical formulation of selective discretization using Zero-Order Hold (ZOH).
- Input-dependent parameter dynamics: $\Delta(x_t), B(x_t), C(x_t)$.
- Linear $O(N)$ sequence processing with constant-size recurrent memory.

## Run
```bash
python 01-selective-scan-ssm/selective_scan.py
```
