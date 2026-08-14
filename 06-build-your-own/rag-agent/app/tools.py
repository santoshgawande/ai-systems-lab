"""Tools the router agent can call.

Each tool is a plain function returning (result_text, metadata). The metadata is
surfaced in the dashboard's RAG inspector so you can see exactly what happened.
"""
from __future__ import annotations

import ast
import operator
import urllib.parse
import urllib.request
import json

from . import vectorstore


def rag_search(query: str) -> tuple[str, dict]:
    """Retrieve relevant chunks from the ingested documents."""
    hits = vectorstore.query(query)
    if not hits:
        return "No relevant documents found.", {"hits": []}
    context = "\n\n".join(
        f"[{i + 1}] (source: {h['source']}, score: {h['score']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return context, {"hits": hits}


# --- safe arithmetic evaluator (no eval()) ---------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> tuple[str, dict]:
    """Evaluate a basic arithmetic expression safely."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result}", {"expression": expression, "result": result}
    except Exception as e:  # noqa: BLE001 - surface any parse error to the user
        return f"Could not evaluate '{expression}': {e}", {"error": str(e)}


def web_search(query: str) -> tuple[str, dict]:
    """Lightweight web lookup via DuckDuckGo's Instant Answer API (no key needed)."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1}
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        answer = data.get("AbstractText") or data.get("Answer") or ""
        if not answer and data.get("RelatedTopics"):
            first = data["RelatedTopics"][0]
            answer = first.get("Text", "") if isinstance(first, dict) else ""
        answer = answer or "No instant answer available."
        return answer, {"query": query, "source": "duckduckgo"}
    except Exception as e:  # noqa: BLE001
        return f"Web search failed: {e}", {"error": str(e)}


# Registry the agent and dashboard both read from.
TOOLS = {
    "rag_search": {
        "fn": rag_search,
        "arg": "query",
        "desc": "Search the ingested knowledge base for relevant document chunks.",
    },
    "calculator": {
        "fn": calculator,
        "arg": "expression",
        "desc": "Evaluate an arithmetic expression, e.g. '12000 * 1.18'.",
    },
    "web_search": {
        "fn": web_search,
        "arg": "query",
        "desc": "Look up a fact on the public web for current/general knowledge.",
    },
}
