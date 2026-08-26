# Lab 02: MTP Native Self-Speculative Decoding

## What You Learn
- Why separate draft models in speculative decoding introduce memory and synchronization overhead.
- Using internal MTP prediction heads as self-draft proposals.
- Validating and correcting multi-token trajectories in a single forward pass.
- Achieving $2\times$ inference speedups with zero extra model weights.

## Run
```bash
python 02-self-speculative-decoding/self_speculative.py
```
