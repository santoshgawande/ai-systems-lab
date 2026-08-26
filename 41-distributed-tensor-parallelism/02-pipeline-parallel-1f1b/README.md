# Lab 02: 1F1B Pipeline Parallelism

## What You Learn
- Why naive GPipe batch execution leads to out-of-memory activation explosions.
- The 1F1B (One-Forward-One-Backward) schedule: Warmup, Steady-State, and Cooldown.
- Capping peak activation memory to $P$ micro-batches.
- Computing pipeline bubble overhead: $F_{bubble} = \frac{P - 1}{M}$.

## Run
```bash
python 02-pipeline-parallel-1f1b/pipeline_1f1b.py
```
