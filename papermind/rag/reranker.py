"""
Cross-encoder reranker using sentence-transformers.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def rerank(query: str, docs: list[dict], top_k: int = 6) -> list[dict]:
    """Score each doc with the cross-encoder and return top_k sorted by score."""
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [(query, doc.get("text", "")[:512]) for doc in docs]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
