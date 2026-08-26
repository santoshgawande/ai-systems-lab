from __future__ import annotations
"""
Test-Time Compute (TTC) & Thinking Budget Engine (DeepSeek-R1 / OpenAI o1/o3).

The paradigm shift in 2025–2026 AI systems is Test-Time Compute Scaling:
Models spend orders-of-magnitude more compute at INFERENCE time generating long,
deliberate internal reasoning chains before emitting the final answer.

Architecture:
- `<think> ... </think>` token enclosure separating internal reasoning from user-visible answer.
- Dynamic Thinking Budget management (enforcing minimum thinking tokens on hard math/coding problems).
- Thought progression & self-correction trigger detection ("Wait, let me rethink", "Alternatively", "Let me check").
"""
import re
from typing import Dict, List, Optional, Tuple
import dataclasses


@dataclasses.dataclass
class ReasoningOutput:
    thinking_content: str
    final_answer: str
    thought_token_count: int
    answer_token_count: int
    num_self_corrections: int
    budget_met: bool


class ThinkingBudgetManager:
    """
    Manages test-time compute thinking tokens and self-correction verification.
    """
    CORRECTION_TRIGGERS = [
        r"\bwait\b",
        r"\blet me double check\b",
        r"\blet me rethink\b",
        r"\balternatively\b",
        r"\bhowever\b",
        r"\bhold on\b",
        r"\bupon closer inspection\b"
    ]

    def __init__(self, min_thought_tokens: int = 10, max_thought_tokens: int = 4096):
        self.min_thought_tokens = min_thought_tokens
        self.max_thought_tokens = max_thought_tokens

    def parse_reasoning_trace(self, raw_llm_response: str) -> ReasoningOutput:
        """
        Extracts <think>...</think> block and measures thinking vs answer tokens.
        """
        think_match = re.search(r"<think>(.*?)</think>", raw_llm_response, re.DOTALL | re.IGNORECASE)
        
        if think_match:
            thinking = think_match.group(1).strip()
            # Remainder after </think> is the final answer
            answer = raw_llm_response[think_match.end():].strip()
        else:
            # If no tags present, entire text is treated as direct response
            thinking = ""
            answer = raw_llm_response.strip()

        thought_tokens = len(thinking.split()) if thinking else 0
        answer_tokens = len(answer.split()) if answer else 0

        # Count self-reflection / backtrack triggers
        num_corrections = 0
        if thinking:
            for pattern in self.CORRECTION_TRIGGERS:
                matches = re.findall(pattern, thinking, re.IGNORECASE)
                num_corrections += len(matches)

        budget_met = thought_tokens >= self.min_thought_tokens

        return ReasoningOutput(
            thinking_content=thinking,
            final_answer=answer,
            thought_token_count=thought_tokens,
            answer_token_count=answer_tokens,
            num_self_corrections=num_corrections,
            budget_met=budget_met
        )

    def estimate_problem_complexity(self, query: str) -> int:
        """
        Dynamically scales thinking budget based on problem complexity.
        Math, logic puzzles, and distributed systems architecture queries require higher compute.
        """
        q_lower = query.lower()
        hard_keywords = ["prove", "solve", "math", "complexity", "concurrency", "distributed", "algorithm", "derive"]
        score = sum(1 for kw in hard_keywords if kw in q_lower)
        if score >= 2:
            return 500  # High compute allocation
        elif score == 1:
            return 150  # Medium compute allocation
        else:
            return 25   # Light compute allocation


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🧠 TEST-TIME COMPUTE & REASONING PARSER (DeepSeek-R1 style) ===\n")

    manager = ThinkingBudgetManager(min_thought_tokens=20)

    sample_r1_output = """<think>
We are asked to find the number of trailing zeroes in 100!.
A trailing zero is created by a factor of 10 = 2 * 5.
In 100!, the count of prime factor 2 is much greater than 5, so we only need to count factors of 5.
Using Legendre's formula:
E_5(100!) = floor(100/5) + floor(100/25) + floor(100/125) ...
100 / 5 = 20.
100 / 25 = 4.
Wait, let me double check if 125 contributes anything: floor(100/125) = 0.
So total = 20 + 4 = 24.
Hold on, let me double check with small example like 10!:
10/5 = 2 trailing zeroes. 10! = 3628800 (two zeroes).
The logic holds up.
</think>
The number of trailing zeroes in 100! is **24**."""

    parsed = manager.parse_reasoning_trace(sample_r1_output)
    print("1. Parsed Reasoning Trace:")
    print(f"   Thought Tokens: {parsed.thought_token_count}")
    print(f"   Answer Tokens:  {parsed.answer_token_count}")
    print(f"   Self-Correction Triggers Found: {parsed.num_self_corrections}")
    print(f"   Thinking Budget Met (>= 20 tokens): {parsed.budget_met}")
    print(f"\n2. Extracted Final Answer:\n{parsed.final_answer}")

    print("\n3. Dynamic Compute Budget Estimation:")
    q1 = "What is the capital of Japan?"
    q2 = "Solve and prove the time complexity of distributed consensus with 2n+1 nodes."
    print(f"   Query: '{q1}' -> Budget: {manager.estimate_problem_complexity(q1)} tokens")
    print(f"   Query: '{q2}' -> Budget: {manager.estimate_problem_complexity(q2)} tokens")
