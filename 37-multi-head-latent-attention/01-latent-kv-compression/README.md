# Lab 01: Multi-Head Latent Attention (MLA) Compression

## What You Learn
- Why standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) struggle with KV-cache memory at long contexts.
- Joint Key-Value low-rank compression into latent vector $c_t^{KV}$.
- Why rotary positional embeddings (RoPE) must be decoupled into a separate small key $k_t^R$.
- Quantifying 93.3% VRAM savings during serving.

## Run
```bash
python 01-latent-kv-compression/mla_compression.py
```
