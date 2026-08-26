# Lab 01: Megatron-LM Tensor Parallelism

## What You Learn
- Why naive tensor slicing requires continuous all-gather synchronization.
- Megatron-LM Column-Parallel ($W_1$) and Row-Parallel ($W_2$) MLP architecture.
- Reducing inter-GPU communication to a single All-Reduce Sum per block.
- Exact numerical equivalence to single-device forward execution.

## Run
```bash
python 01-column-row-parallelism/tensor_parallel.py
```
