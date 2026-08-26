# Lab 01: Online Softmax Algorithm

## What You Learn
- Why standard 3-pass softmax causes memory-bandwidth bottlenecks in GPU attention.
- The mathematical derivation of the 1-pass online softmax rescaling formula:
  $$m_i = \max(m_{i-1}, x_i), \quad d_i = d_{i-1} e^{m_{i-1}-m_i} + e^{x_i-m_i}$$
- How chunked block updates form the mathematical foundation of FlashAttention.

## Run
```bash
python 01-online-softmax/online_softmax.py
```
