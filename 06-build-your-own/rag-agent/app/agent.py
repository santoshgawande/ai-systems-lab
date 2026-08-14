"""Router agent: decides which tool to use, runs it, then synthesizes an answer.

The flow (and why it's good for learning):
  1. ROUTE  - LLM picks a tool + argument as JSON. This is "function routing"
              done in a provider-agnostic way (works on Claude *and* Ollama).
  2. ACT    - we execute the chosen tool in Python (the LLM never runs code).
  3. ANSWER - LLM writes the final reply grounded in the tool's output.

Every step is captured in a trace so the dashboard can show the agent's reasoning.
"""
from __future__ import annotations

from .llm import get_llm
from .tools import TOOLS

ROUTER_SYSTEM = """You are a routing agent. Given the user's question, choose exactly one tool to help answer it.

Available tools:
{tool_list}

Respond with ONLY a JSON object, no prose, in this exact shape:
{{"tool": "<tool_name>", "arg": "<the single string argument for that tool>", "reason": "<one short sentence>"}}

Guidance:
- Use rag_search for questions about the user's documents/knowledge base.
- Use calculator for arithmetic.
- Use web_search for current events or general facts not in the documents.
"""

ANSWER_SYSTEM = """You are a helpful assistant. Answer the user's question using the TOOL RESULT below.
Be concise. If the tool result is document context, ground your answer in it and cite sources like [1], [2].
If the tool result does not contain the answer, say so honestly."""


def _tool_list() -> str:
    return "\n".join(f"- {name}: {meta['desc']}" for name, meta in TOOLS.items())


def route(question: str) -> dict:
    """Step 1: ask the LLM which tool to use."""
    llm = get_llm()
    system = ROUTER_SYSTEM.format(tool_list=_tool_list())
    try:
        decision = llm.complete_json(question, system=system)
    except Exception:
        # If the model gives unparseable output, default to RAG — the safe choice.
        decision = {"tool": "rag_search", "arg": question, "reason": "router fallback"}
    if decision.get("tool") not in TOOLS:
        decision = {"tool": "rag_search", "arg": question, "reason": "unknown tool -> rag"}
    return decision


def run(question: str) -> dict:
    """Full route -> act -> answer cycle. Returns the answer plus a full trace."""
    llm = get_llm()

    decision = route(question)
    tool_name = decision["tool"]
    arg = decision.get("arg", question)

    tool_output, tool_meta = TOOLS[tool_name]["fn"](arg)

    answer_prompt = (
        f"User question: {question}\n\n"
        f"TOOL RESULT (from {tool_name}):\n{tool_output}\n\n"
        "Now write the final answer."
    )
    answer = llm.complete(answer_prompt, system=ANSWER_SYSTEM)

    return {
        "answer": answer,
        "trace": {
            "provider": llm.name,
            "decision": decision,
            "tool": tool_name,
            "tool_arg": arg,
            "tool_meta": tool_meta,
            "tool_output": tool_output,
        },
    }
