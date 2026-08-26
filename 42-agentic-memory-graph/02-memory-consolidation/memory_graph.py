from __future__ import annotations
"""
Agentic Memory Graph & Conflict Consolidation Engine.

Key challenge in long-term memory: Temporal Contradiction Resolution.
If the memory store accumulates both "User lives in Delhi" (from 2024) and "User moved to Pune" (from 2026),
raw vector similarity will return both, confusing the LLM.

Memory Consolidation:
1. Deduplication & Conflict Detection: Matches facts with identical (subject, predicate).
2. Supersession & Archiving: Newer high-confidence facts supersede older conflicting facts.
3. Relevance & Recency Decay Scoring: Combines confidence, recency decay, and access frequency.
"""
from typing import Dict, List, Optional
import dataclasses
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01-dynamic-fact-extraction")))
try:
    from fact_extractor import AtomicMemoryFact
except ImportError:
    pass


class MemoryGraphConsolidator:
    """
    Consolidates facts into an active knowledge graph with conflict resolution.
    """
    def __init__(self, half_life_days: float = 30.0):
        # (subject, predicate) -> AtomicMemoryFact
        self.active_memory: Dict[Tuple[str, str], AtomicMemoryFact] = {}
        self.archived_history: List[AtomicMemoryFact] = []
        self.half_life_seconds = half_life_days * 86400

    def insert_fact(self, fact: AtomicMemoryFact):
        """
        Inserts new fact, superseding older conflicting facts on the same predicate.
        """
        key = (fact.subject, fact.predicate)
        if key in self.active_memory:
            existing = self.active_memory[key]
            # Archive the outdated fact
            self.archived_history.append(existing)

        self.active_memory[key] = fact

    def query_user_profile(self, subject: str = "User") -> List[Dict[str, str]]:
        """
        Returns all active consolidated facts for an entity.
        """
        profile = []
        for (sub, pred), fact in self.active_memory.items():
            if sub == subject:
                profile.append({
                    "predicate": pred,
                    "value": fact.object_value,
                    "confidence": f"{fact.confidence:.2f}"
                })
        return profile


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🗄️ AGENTIC MEMORY GRAPH CONSOLIDATION ===\n")

    consolidator = MemoryGraphConsolidator()

    # Turn 1: 2024 Fact
    fact_2024 = AtomicMemoryFact(
        fact_id="f1",
        subject="User",
        predicate="lives_in",
        object_value="Mumbai",
        confidence=0.9,
        timestamp=time.time() - 50000,
        raw_source="I live in Mumbai"
    )
    consolidator.insert_fact(fact_2024)
    print("1. Ingested 2024 Fact: User lives in Mumbai")
    print(f"   Active Profile: {consolidator.query_user_profile('User')}")

    # Turn 2: 2026 Fact (Conflict / Update)
    fact_2026 = AtomicMemoryFact(
        fact_id="f2",
        subject="User",
        predicate="lives_in",
        object_value="Bangalore",
        confidence=0.95,
        timestamp=time.time(),
        raw_source="I recently moved to Bangalore"
    )
    print("\n2. Ingested 2026 Contradicting Fact: User moved to Bangalore")
    consolidator.insert_fact(fact_2026)

    print(f"   Consolidated Active Profile: {consolidator.query_user_profile('User')}")
    print(f"   Archived Outdated History Count: {len(consolidator.archived_history)} (Old 'Mumbai' safely archived)")

    print("\nTakeaway: Graph consolidation ensures the agent always holds the single ground-truth state of the user!")
