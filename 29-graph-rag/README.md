# Section 29 — GraphRAG (Knowledge Graph Augmented Retrieval)

Bridge the gap between granular entity relationships and global dataset-wide summarization.

## What you learn

- Why Vector RAG fails on multi-hop and holistic global queries
- Knowledge graph extraction: Entities, Relations, Descriptions
- Community detection (Leiden/Louvain clustering)
- Hierarchical community reports (Level 0 Micro -> Level 1 Macro)
- Dual-mode GraphRAG retrieval: Local Entity Search vs Global Map-Reduce Search

## Labs

| Lab | What it covers |
|---|---|
| 01-knowledge-graph-extraction | Entity & Relationship triplet extraction, graph traversal |
| 02-community-detection-hierarchical-rag | Graph clustering, community summaries, Global vs Local search |

## Setup

```bash
python 01-knowledge-graph-extraction/kg_extractor.py
python 02-community-detection-hierarchical-rag/graph_rag.py
```
