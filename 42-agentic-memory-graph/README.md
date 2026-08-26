# 42. Agentic Memory Graphs

Agentic memory architectures (Mem0, Zep, Letta) enable AI assistants to maintain persistent, evolving knowledge across sessions while automatically resolving temporal conflicts and contradictions.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-dynamic-fact-extraction` | Dynamic Fact & Relationship Extraction | Entity-predicate extraction, temporal timestamps |
| `02-memory-consolidation` | Memory Graph Consolidation | Contradiction resolution, supersession, state profiles |

## Key Concepts

- **Beyond Vector RAG**: Pure vector search over raw transcript history returns contradictory and redundant statements. Memory graphs maintain an evolving ground-truth state of the user.
- **Declarative Memory**: Separating user preferences, episodic experiences, and semantic facts into structured atomic graph nodes.
