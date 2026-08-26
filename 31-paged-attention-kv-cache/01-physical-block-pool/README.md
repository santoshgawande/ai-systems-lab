# Lab 01: PagedAttention Physical Block Pool

## What You Learn
- Why contiguous KV-cache allocation causes 60–80% GPU memory fragmentation.
- How PagedAttention applies virtual memory paging principles to LLM KV-caches.
- Managing Physical Blocks ($B=16$), Logical-to-Physical Block Tables, and dynamic slot allocation.
- Deterministic memory reclamation upon sequence termination.

## Run
```bash
python 01-physical-block-pool/block_allocator.py
```
