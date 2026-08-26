# Lab 02: Radix-Tree KV-Cache Prefix Caching

## What You Learn
- How RadixAttention stores past KV-caches in a Radix Tree index.
- Prefix matching for multi-turn chats and shared system instructions.
- Eliminating Time-to-First-Token (TTFT) prefill compute on cached prefixes.
- Calculating global token cache hit rates.

## Run
```bash
python 02-prefix-caching-radix/prefix_cache.py
```
