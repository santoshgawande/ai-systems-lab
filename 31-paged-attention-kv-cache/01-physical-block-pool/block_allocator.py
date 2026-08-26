from __future__ import annotations
"""
PagedAttention Physical Block Memory Allocator (Kwon et al., SOSP 2023 / vLLM).

Standard LLM serving pre-allocates contiguous memory for maximum context length (e.g. 8k tokens),
wasting 60–80% of GPU VRAM to internal and external memory fragmentation.

PagedAttention operates like OS Virtual Memory paging:
- Divides KV-cache into fixed-size Physical Blocks (e.g., 16 tokens/block).
- Maintains a Logical-to-Physical Block Table per sequence.
- Allocates new physical blocks on demand as new tokens are generated.
- Eliminates external memory fragmentation (near 100% memory utilization).
"""
from typing import Dict, List, Optional, Set
import dataclasses


@dataclasses.dataclass
class PhysicalBlock:
    block_id: int
    block_size: int
    tokens: List[str] = dataclasses.field(default_factory=list)
    ref_count: int = 0

    @property
    def is_full(self) -> bool:
        return len(self.tokens) >= self.block_size

    @property
    def num_free_slots(self) -> int:
        return self.block_size - len(self.tokens)

    def append_token(self, token: str) -> bool:
        if self.is_full:
            return False
        self.tokens.append(token)
        return True

    def clear(self):
        self.tokens.clear()
        self.ref_count = 0


class PagedAttentionBlockAllocator:
    """
    Manages physical GPU memory block pools and per-sequence block tables.
    """
    def __init__(self, num_blocks: int = 64, block_size: int = 4):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        }
        # seq_id -> list of physical block IDs
        self.block_tables: Dict[str, List[int]] = {}

    def allocate_block(self) -> Optional[int]:
        if not self.free_blocks:
            return None
        block_id = self.free_blocks.pop(0)
        self.blocks[block_id].ref_count = 1
        return block_id

    def free_block(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.clear()
            if block_id not in self.free_blocks:
                self.free_blocks.append(block_id)

    def register_sequence(self, seq_id: str, prompt_tokens: List[str]) -> bool:
        """
        Initializes block table for a prompt sequence.
        """
        self.block_tables[seq_id] = []
        for token in prompt_tokens:
            if not self.append_token_to_seq(seq_id, token):
                return False
        return True

    def append_token_to_seq(self, seq_id: str, token: str) -> bool:
        """
        Appends 1 token to sequence, allocating a new physical block if current block is full.
        """
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []

        table = self.block_tables[seq_id]
        if not table or self.blocks[table[-1]].is_full:
            # Need a new physical block
            new_block_id = self.allocate_block()
            if new_block_id is None:
                return False  # Out of memory
            table.append(new_block_id)

        current_block = self.blocks[table[-1]]
        current_block.append_token(token)
        return True

    def free_sequence(self, seq_id: str):
        """Frees all physical blocks allocated to the sequence."""
        if seq_id in self.block_tables:
            for b_id in self.block_tables[seq_id]:
                self.free_block(b_id)
            del self.block_tables[seq_id]

    def get_sequence_tokens(self, seq_id: str) -> List[str]:
        tokens = []
        for b_id in self.block_tables.get(seq_id, []):
            tokens.extend(self.blocks[b_id].tokens)
        return tokens

    @property
    def memory_utilization(self) -> float:
        """Calculates actual tokens stored vs total memory capacity."""
        total_slots = self.num_blocks * self.block_size
        used_slots = sum(len(b.tokens) for b in self.blocks.values() if b.ref_count > 0)
        return used_slots / total_slots if total_slots > 0 else 0.0


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 📦 PAGEDATTENTION PHYSICAL BLOCK ALLOCATOR (vLLM) ===\n")

    allocator = PagedAttentionBlockAllocator(num_blocks=8, block_size=4)
    print(f"Initialized allocator with {allocator.num_blocks} blocks of size {allocator.block_size} (Capacity: 32 tokens)")

    print("\n1. Ingesting prompt for Sequence 1: 'The capital of France is Paris and'")
    s1_prompt = ["The", "capital", "of", "France", "is", "Paris", "and"]
    allocator.register_sequence("seq_1", s1_prompt)
    print(f"   Seq 1 Block Table: {allocator.block_tables['seq_1']}")
    for b_id in allocator.block_tables['seq_1']:
        print(f"     Physical Block [{b_id}]: {allocator.blocks[b_id].tokens}")

    print("\n2. Generating new tokens for Seq 1 autoregressively...")
    for tok in ["it", "has", "many", "museums", "."]:
        allocator.append_token_to_seq("seq_1", tok)
        print(f"   Generated '{tok}' -> Seq 1 Table: {allocator.block_tables['seq_1']}")

    print(f"\nSeq 1 Full Context: {' '.join(allocator.get_sequence_tokens('seq_1'))}")
    print(f"Current Memory Utilization: {allocator.memory_utilization:.1%}")
    print(f"Free Blocks remaining: {len(allocator.free_blocks)} / {allocator.num_blocks}")

    print("\n3. Sequence 1 finishes -> Freeing sequence blocks...")
    allocator.free_sequence("seq_1")
    print(f"Free Blocks after cleanup: {len(allocator.free_blocks)} / {allocator.num_blocks} (100% reclaimed)")
