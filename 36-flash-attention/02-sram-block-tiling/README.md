# Lab 02: FlashAttention SRAM Block Tiling

## What You Learn
- Why standard attention is memory-bandwidth bound rather than compute bound.
- The GPU memory hierarchy: on-chip SRAM (19 TB/s) vs High Bandwidth Memory (HBM, 2 TB/s).
- Tiling Query ($B_r$) and Key/Value ($B_c$) blocks directly into SRAM.
- Rescaling output accumulators with running statistics without writing $N \times N$ attention matrices to HBM.

## Run
```bash
python 02-sram-block-tiling/flash_attention.py
```
