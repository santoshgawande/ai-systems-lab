# 31. PagedAttention & KV-Cache Management

Managing Key-Value (KV) cache memory is the single most critical engineering challenge in high-throughput LLM serving. PagedAttention and Radix Prefix Caching eliminate memory waste and unlock massive batch sizes.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-physical-block-pool` | PagedAttention Block Allocator | Physical block pool, block tables, non-contiguous VRAM allocation |
| `02-prefix-caching-radix` | Radix Prefix Caching | Radix tree indexing, prefix sharing, sub-millisecond TTFT |

## Key Concepts

- **Virtual Memory for GPUs**: Just as OS paging stops process fragmentation, PagedAttention decouples logical sequence length from physical memory allocation.
- **Copy-on-Write (COW)**: Enables beam search and parallel sampling with zero memory duplication until paths diverge.
- **Prompt Prefix Reuse**: System prompts and multi-turn message prefixes are matched and reused directly from GPU SRAM/HBM without re-computation.
