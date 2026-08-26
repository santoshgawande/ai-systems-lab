# Lab 02: MLA Matrix Absorption

## What You Learn
- Why decompressing stored Keys during inference wastes GPU memory bandwidth.
- Mathematical proof of Matrix Absorption: $(q^T W^{UK}) c^{KV} = (W^{UK^T} q)^T c^{KV}$.
- Pre-projecting queries into latent dimension $d_c$ to perform direct dot products against cached latent vectors.
- Reducing inference GEMM operations to minimum compute.

## Run
```bash
python 02-matrix-absorption/matrix_absorption.py
```
