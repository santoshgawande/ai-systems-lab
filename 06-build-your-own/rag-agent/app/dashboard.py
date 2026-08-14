"""Streamlit dashboard: chat with the agent + inspect what RAG retrieved.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import streamlit as st

from app import ingest, vectorstore
from app.agent import run
from app.config import config

st.set_page_config(page_title="RAG Agent Lab", page_icon="🔎", layout="wide")

# --- sidebar: status + ingestion controls ---------------------------------
with st.sidebar:
    st.title("🔎 RAG Agent Lab")
    st.caption("RAG pipeline · router agent · live inspector")

    st.subheader("Backend")
    st.write(f"**LLM provider:** `{config.llm_provider}`")
    st.write(
        f"**Model:** `{config.anthropic_model if config.llm_provider == 'anthropic' else config.ollama_model}`"
    )
    st.write(f"**Embeddings:** `{config.embedding_model}`")

    st.subheader("Knowledge base")
    try:
        st.metric("Chunks indexed", vectorstore.stats()["chunks"])
    except Exception as e:  # noqa: BLE001
        st.warning(f"Store not ready: {e}")

    if st.button("📥 Ingest data/docs", use_container_width=True):
        with st.spinner("Chunking + embedding..."):
            report = ingest.ingest_all()
        st.success("Ingested: " + ", ".join(f"{k} ({v})" for k, v in report.items()))
        st.rerun()

# --- main: two columns, chat on the left, inspector on the right -----------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, result-dict)

chat_col, inspect_col = st.columns([3, 2])

with chat_col:
    st.header("Chat")
    for q, result in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(result["answer"])

    question = st.chat_input("Ask something about your documents, math, or the web...")
    if question:
        with st.spinner("Routing + answering..."):
            result = run(question)
        st.session_state.history.append((question, result))
        st.rerun()

with inspect_col:
    st.header("🔬 Inspector")
    if not st.session_state.history:
        st.info("Ask a question to see the agent's routing decision and retrieved chunks.")
    else:
        trace = st.session_state.history[-1][1]["trace"]
        st.write(f"**Provider:** `{trace['provider']}`")
        st.write(f"**Tool chosen:** `{trace['tool']}`")
        st.caption(f"Reason: {trace['decision'].get('reason', '—')}")
        st.code(f"arg = {trace['tool_arg']!r}", language="python")

        if trace["tool"] == "rag_search":
            hits = trace["tool_meta"].get("hits", [])
            st.subheader(f"Retrieved chunks ({len(hits)})")
            for i, h in enumerate(hits, 1):
                with st.expander(f"[{i}] {h['source']}  ·  score {h['score']}"):
                    st.write(h["text"])
        else:
            st.subheader("Tool output")
            st.code(trace["tool_output"])
            st.json(trace["tool_meta"])
