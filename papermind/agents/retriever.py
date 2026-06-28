"""
Retriever agent — runs hybrid retrieval (dense + BM25) and
returns the top reranked documents.
"""
from __future__ import annotations

from typing import Any

from papermind.rag.retrieval import HybridRetriever
from papermind.state import ResearchState

TOP_K = 6


def run_retriever(state: ResearchState, retriever: HybridRetriever) -> dict[str, Any]:
    query = state.get("retrieval_query") or state["query"]
    docs = retriever.search(query, top_k=TOP_K)
    return {"retrieved_docs": docs}
