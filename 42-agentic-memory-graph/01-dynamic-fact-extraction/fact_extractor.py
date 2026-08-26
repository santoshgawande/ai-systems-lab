from __future__ import annotations
"""
Dynamic Fact Extraction & Long-Term Memory Engine (Mem0 / Zep / Letta).

LLMs lose conversation history across sessions. Simple vector similarity search over raw transcripts
returns noisy, redundant context and fails to resolve temporal contradictions.

Dynamic Fact Extraction converts conversational turns into atomic declarative knowledge:
  User: "I just relocated from Mumbai to Bangalore to join a new fintech startup."
  Extracted Atomic Facts:
    1. User.location = "Bangalore" (supersedes "Mumbai")
    2. User.industry = "Fintech"
    3. User.role = "Startup Employee"
"""
import re
from typing import Dict, List, Optional
import dataclasses
import time


@dataclasses.dataclass
class AtomicMemoryFact:
    fact_id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float
    timestamp: float
    raw_source: str


class DynamicFactExtractor:
    """
    Extracts structured atomic facts and user preferences from dialogue turns.
    """
    def __init__(self):
        self._fact_counter = 0

    def extract_facts_from_turn(self, user_message: str) -> List[AtomicMemoryFact]:
        """
        Parses text and extracts structured relational facts.
        """
        facts: List[AtomicMemoryFact] = []
        now = time.time()
        msg_lower = user_message.lower()

        # Rule 1: Location Extraction
        loc_match = re.search(r"(?:moved to|living in|relocated to|located in|live in)\s+([A-Za-z\s]+?)(?:\.|\,|$|to|and)", user_message, re.IGNORECASE)
        if loc_match:
            self._fact_counter += 1
            loc = loc_match.group(1).strip()
            facts.append(AtomicMemoryFact(
                fact_id=f"fact_{self._fact_counter}",
                subject="User",
                predicate="lives_in",
                object_value=loc,
                confidence=0.95,
                timestamp=now,
                raw_source=user_message
            ))

        # Rule 2: Programming / Tech Preferences
        pref_match = re.search(r"(?:prefer|like|use|write in)\s+(python|typescript|golang|rust|java|c\+\+)", user_message, re.IGNORECASE)
        if pref_match:
            self._fact_counter += 1
            lang = pref_match.group(1).capitalize()
            facts.append(AtomicMemoryFact(
                fact_id=f"fact_{self._fact_counter}",
                subject="User",
                predicate="prefers_language",
                object_value=lang,
                confidence=0.90,
                timestamp=now,
                raw_source=user_message
            ))

        # Rule 3: Workplace / Organization
        work_match = re.search(r"(?:work at|joined|employed at|at)\s+([A-Z][A-Za-z0-9\s]+?)(?:\.|\,|$)", user_message)
        if work_match:
            self._fact_counter += 1
            org = work_match.group(1).strip()
            facts.append(AtomicMemoryFact(
                fact_id=f"fact_{self._fact_counter}",
                subject="User",
                predicate="works_at",
                object_value=org,
                confidence=0.88,
                timestamp=now,
                raw_source=user_message
            ))

        return facts


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🧠 DYNAMIC AGENTIC FACT EXTRACTION (Mem0 / Zep) ===\n")

    extractor = DynamicFactExtractor()

    sample_dialogues = [
        "Hi! I recently moved to Bangalore and I prefer Python for all backend development.",
        "I just joined Zerodha to work on their low-latency order matching switch."
    ]

    for turn_idx, msg in enumerate(sample_dialogues, 1):
        print(f"Turn {turn_idx}: '{msg}'")
        extracted = extractor.extract_facts_from_turn(msg)
        print(f"  Extracted {len(extracted)} Atomic Facts:")
        for f in extracted:
            print(f"    • [{f.fact_id}] ({f.subject}) --[{f.predicate}]--> '{f.object_value}' (conf={f.confidence:.2f})")
        print()

    print("Takeaway: Converting conversational prose into structured atomic facts eliminates context bloat and temporal confusion!")
