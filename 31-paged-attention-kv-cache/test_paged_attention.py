import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "01-physical-block-pool"))
sys.path.insert(0, os.path.join(base_dir, "02-prefix-caching-radix"))

from block_allocator import PagedAttentionBlockAllocator
from prefix_cache import RadixPrefixCache


class TestPagedAttention(unittest.TestCase):
    def test_paged_attention_allocation_and_cleanup(self):
        allocator = PagedAttentionBlockAllocator(num_blocks=4, block_size=2)
        
        # Ingest 3 tokens -> needs 2 physical blocks (capacity 2 each)
        success = allocator.register_sequence("s1", ["tok1", "tok2", "tok3"])
        self.assertTrue(success)
        self.assertEqual(len(allocator.block_tables["s1"]), 2)
        self.assertEqual(len(allocator.free_blocks), 2)
        
        tokens = allocator.get_sequence_tokens("s1")
        self.assertEqual(tokens, ["tok1", "tok2", "tok3"])
        
        # Free sequence
        allocator.free_sequence("s1")
        self.assertEqual(len(allocator.free_blocks), 4)

    def test_radix_prefix_cache_matching(self):
        cache = RadixPrefixCache(block_size=2)
        prompt1 = ["Sys", "Prompt", "A", "B"]
        cache.insert_sequence(prompt1)
        
        prompt2 = ["Sys", "Prompt", "C", "D"]
        blocks, matched = cache.match_prefix(prompt2)
        # Should match "Sys", "Prompt" (length 2)
        self.assertEqual(matched, 2)
        self.assertGreater(cache.cache_hit_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
