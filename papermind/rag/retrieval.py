"""
HybridRetriever: combines Pinecone dense search with BM25 and reranks results.

The BM25 index is built lazily from the Pinecone metadata corpus on first call.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from papermind.rag.reranker import rerank
from papermind.rag.ingest import EMBEDDING_MODEL

DENSE_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 6


def _embed_query(text: str) -> list[float]:
    client = OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


class HybridRetriever:
    def __init__(self, index_name: str | None = None):
        api_key = os.environ.get("PINECONE_API_KEY", "")
        index_name = index_name or os.environ.get("PINECONE_INDEX", "papermind")

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # BM25 index (built lazily from Pinecone metadata)
    # ------------------------------------------------------------------

    def _build_bm25(self) -> None:
        """Fetch a sample of vectors from Pinecone to build the BM25 corpus."""
        # list() returns paginated vector IDs; we fetch metadata via fetch()
        try:
            id_page = self._index.list(limit=500)
            ids = list(id_page)
            if not ids:
                self._corpus = []
                self._bm25 = None
                return

            fetch_result = self._index.fetch(ids=ids[:500])
            docs = []
            for vec_id, vec in fetch_result.vectors.items():
                meta = vec.metadata or {}
                docs.append(
                    {
                        "id": vec_id,
                        "text": meta.get("text", ""),
                        "metadata": meta,
                    }
                )
            self._corpus = docs
            tokenized = [doc["text"].lower().split() for doc in docs]
            self._bm25 = BM25Okapi(tokenized)
        except Exception:
            self._corpus = []
            self._bm25 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[dict[str, Any]]:
        embedding = _embed_query(query)
        result = self._index.query(vector=embedding, top_k=top_k, include_metadata=True)
        docs = []
        for match in result.matches:
            meta = match.metadata or {}
            docs.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "text": meta.get("text", ""),
                    "metadata": meta,
                }
            )
        return docs

    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict[str, Any]]:
        if self._bm25 is None:
            self._build_bm25()
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            doc = self._corpus[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)
        return results

    def search(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict[str, Any]]:
        """Hybrid search: merge dense + BM25 candidates, then rerank."""
        dense_docs = self.dense_search(query, top_k=DENSE_TOP_K)
        bm25_docs = self.bm25_search(query, top_k=BM25_TOP_K)

        # Deduplicate by id, preferring higher dense score
        seen: dict[str, dict[str, Any]] = {}
        for doc in dense_docs:
            seen[doc["id"]] = doc
        for doc in bm25_docs:
            if doc["id"] not in seen:
                seen[doc["id"]] = doc

        merged = list(seen.values())
        return rerank(query, merged, top_k=top_k)
