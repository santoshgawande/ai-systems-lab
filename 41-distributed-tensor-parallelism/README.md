# 41. Distributed Tensor & Pipeline Parallelism

Distributed 3D Parallelism (Megatron-LM & DeepSpeed) is the engineering foundation enabling the training and serving of 70B+ and 405B+ parameter LLMs across multi-GPU cluster topologies.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-column-row-parallelism` | Megatron Column-Row Tensor Parallelism | Weight sharding, zero intermediate sync, single All-Reduce |
| `02-pipeline-parallel-1f1b` | 1F1B Pipeline Parallelism | Layer pipelining, activation memory cap, micro-batch scheduling |

## Key Concepts

- **Tensor Parallelism (TP)**: Shards individual layer matrices within a single server node over high-speed NVLink (900 GB/s).
- **Pipeline Parallelism (PP)**: Partitions sequential model layers across distinct server nodes over InfiniBand networks.
- **Activation Memory Management**: 1F1B interleaves forward and backward passes to prevent activation accumulation.
