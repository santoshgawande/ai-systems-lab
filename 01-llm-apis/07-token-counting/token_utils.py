"""
Shared token counting utilities.

Uses tiktoken (cl100k_base) as the reference tokenizer — same family used by
GPT-4 and close enough to Claude/Llama for learning purposes. Real production
systems use model-specific tokenizers; the counts will differ slightly but the
*overhead patterns* are identical.
"""

import json
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def count_tokens_in_messages(messages: list[dict]) -> int:
    """Count tokens across an OpenAI-style message list (role + content)."""
    total = 0
    for msg in messages:
        # 4 tokens overhead per message (role framing)
        total += 4
        total += count_tokens(msg.get("content") or "")
        if "tool_calls" in msg:
            total += count_tokens(json.dumps(msg["tool_calls"]))
    return total


def count_tokens_in_tools(tools: list[dict]) -> int:
    """Count tokens that tool definitions add to the context."""
    return count_tokens(json.dumps(tools))


def token_report(label: str, input_tokens: int, output_tokens: int,
                 note: str = "") -> dict:
    """Return a structured cost report dict."""
    # Approximate Claude Sonnet pricing (per-million tokens, as of 2025)
    INPUT_COST_PER_M = 3.0
    OUTPUT_COST_PER_M = 15.0

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    total_cost = input_cost + output_cost

    return {
        "label": label,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8),
        "note": note,
    }
