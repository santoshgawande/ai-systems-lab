from __future__ import annotations
"""
Radix-Tree KV-Cache Prefix Caching Engine (SGLang RadixAttention & Anthropic Prompt Caching).

Shared system prompts, multi-turn chat history, and few-shot examples frequently repeat
across requests.
RadixAttention retains the KV-cache of completed requests in a Radix Tree.
Subsequent requests with matching token prefixes skip KV computation entirely:
  - Cache Hit: 0 prompt computation FLOPs, instant Time-to-First-Token (TTFT).
  - Copy-on-Write (COW): Multiple requests share identical physical KV blocks.
"""
from typing import Dict, List, Optional, Tuple
import dataclasses


@dataclasses.dataclass
class RadixNode:
    token_chunk: List[str]
    physical_block_ids: List[int]
    children: Dict[str, RadixNode] = dataclasses.field(default_factory=dict)
    ref_count: int = 1

    @property
    def key_token(self) -> str:
        return self.token_chunk[0] if self.token_chunk else ""


class RadixPrefixCache:
    """
    Radix tree data structure indexing cached KV-blocks by token prefixes.
    """
    def __init__(self, block_size: int = 4):
        self.block_size = block_size
        self.root = RadixNode(token_chunk=[], physical_block_ids=[])
        self.total_queries = 0
        self.total_tokens_queried = 0
        self.total_tokens_cached = 0
        self.next_block_id = 0

    def _allocate_mock_blocks(self, count: int) -> List[int]:
        blocks = list(range(self.next_block_id, self.next_block_id + count))
        self.next_block_id += count
        return blocks

    def _common_prefix_len(self, list_a: List[str], list_b: List[str]) -> int:
        idx = 0
        while idx < len(list_a) and idx < len(list_b) and list_a[idx] == list_b[idx]:
            idx += 1
        return idx

    def match_prefix(self, tokens: List[str]) -> Tuple[List[int], int]:
        """
        Finds the longest cached prefix matching the input tokens.
        Returns:
            (cached_block_ids, num_cached_tokens)
        """
        self.total_queries += 1
        self.total_tokens_queried += len(tokens)

        matched_blocks: List[int] = []
        matched_tokens = 0
        curr_node = self.root
        curr_pos = 0

        while curr_pos < len(tokens):
            next_token = tokens[curr_pos]
            if next_token not in curr_node.children:
                break

            child = curr_node.children[next_token]
            remaining_tokens = tokens[curr_pos:]
            common_len = self._common_prefix_len(child.token_chunk, remaining_tokens)

            if common_len == len(child.token_chunk):
                # Full child match, descend
                matched_blocks.extend(child.physical_block_ids)
                matched_tokens += common_len
                curr_pos += common_len
                curr_node = child
            elif common_len > 0:
                # Partial match inside this child node
                matched_tokens += common_len
                curr_pos += common_len
                break
            else:
                break

        self.total_tokens_cached += matched_tokens
        return matched_blocks, matched_tokens

    def insert_sequence(self, tokens: List[str]) -> List[int]:
        """
        Inserts tokens into the Radix Tree, splitting nodes if necessary and allocating blocks.
        """
        if not tokens:
            return []

        curr_node = self.root
        curr_pos = 0
        all_blocks: List[int] = []

        while curr_pos < len(tokens):
            next_token = tokens[curr_pos]

            if next_token not in curr_node.children:
                # Insert entire remaining suffix as new child
                unmatched_tokens = tokens[curr_pos:]
                num_blocks = (len(unmatched_tokens) + self.block_size - 1) // self.block_size
                new_blocks = self._allocate_mock_blocks(num_blocks)
                all_blocks.extend(new_blocks)

                new_child = RadixNode(
                    token_chunk=unmatched_tokens,
                    physical_block_ids=new_blocks
                )
                curr_node.children[next_token] = new_child
                break

            child = curr_node.children[next_token]
            remaining = tokens[curr_pos:]
            common_len = self._common_prefix_len(child.token_chunk, remaining)

            if common_len == len(child.token_chunk):
                # Full match of child chunk
                all_blocks.extend(child.physical_block_ids)
                curr_pos += common_len
                curr_node = child
            else:
                # Node split required!
                # Split child.token_chunk at common_len
                split_head = child.token_chunk[:common_len]
                split_tail = child.token_chunk[common_len:]

                head_blocks = child.physical_block_ids  # Reused by head
                # Create branch child for the tail
                tail_child = RadixNode(
                    token_chunk=split_tail,
                    physical_block_ids=self._allocate_mock_blocks(1),
                    children=child.children
                )

                # Mutate current child into head node
                child.token_chunk = split_head
                child.children = {split_tail[0]: tail_child}

                all_blocks.extend(child.physical_block_ids)
                curr_pos += common_len
                curr_node = child

        return all_blocks

    @property
    def cache_hit_rate(self) -> float:
        if self.total_tokens_queried == 0:
            return 0.0
        return self.total_tokens_cached / self.total_tokens_queried


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🌲 RADIX-TREE PREFIX KV-CACHE (SGLang / RadixAttention) ===\n")

    cache = RadixPrefixCache(block_size=4)
    system_prompt = ["You", "are", "a", "helpful", "coding", "assistant", "."]

    print("1. Request 1 arrives with system prompt + Query A:")
    req1 = system_prompt + ["How", "to", "sort", "list", "?"]
    blocks1 = cache.insert_sequence(req1)
    print(f"   Request 1 allocated blocks: {blocks1} (Tokens: {len(req1)})")

    print("\n2. Request 2 arrives sharing the identical system prompt + Query B:")
    req2 = system_prompt + ["Explain", "quicksort", "in", "Python"]
    cached_blocks, hit_tokens = cache.match_prefix(req2)
    print(f"   Cache Hit: {hit_tokens} / {len(req2)} tokens matched ({hit_tokens/len(req2):.1%})")
    print(f"   Reused Prefix Tokens: {req2[:hit_tokens]}")
    blocks2 = cache.insert_sequence(req2)
    print(f"   Total allocated blocks for Req 2: {blocks2}")

    print("\n3. Request 3 (Multi-turn turn 2 on Request 1):")
    req3 = req1 + ["Provide", "an", "example"]
    cached_blocks3, hit_tokens3 = cache.match_prefix(req3)
    print(f"   Cache Hit: {hit_tokens3} / {len(req3)} tokens matched ({hit_tokens3/len(req3):.1%})")

    print(f"\nOverall Cache Hit Rate across all requests: {cache.cache_hit_rate:.1%}")
    print("Takeaway: Prefix Caching slashes prompt compute cost & TTFT latency to near zero for repeated system prompts!")
