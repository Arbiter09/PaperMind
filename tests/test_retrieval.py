"""Unit tests for HybridRetriever and reranker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    def test_rerank_returns_top_k(self):
        with patch("papermind.rag.reranker.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.1]
            mock_get.return_value = mock_model

            from papermind.rag.reranker import rerank

            docs = [
                {"text": "doc A", "metadata": {}},
                {"text": "doc B", "metadata": {}},
                {"text": "doc C", "metadata": {}},
                {"text": "doc D", "metadata": {}},
            ]
            result = rerank("test query", docs, top_k=2)

            assert len(result) == 2
            assert result[0]["text"] == "doc A"  # score 0.9
            assert result[1]["text"] == "doc C"  # score 0.7

    def test_rerank_empty_docs(self):
        from papermind.rag.reranker import rerank
        result = rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_fewer_docs_than_top_k(self):
        with patch("papermind.rag.reranker.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.5, 0.8]
            mock_get.return_value = mock_model

            from papermind.rag.reranker import rerank

            docs = [{"text": "x", "metadata": {}}, {"text": "y", "metadata": {}}]
            result = rerank("q", docs, top_k=10)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_retriever(self):
        """Build a HybridRetriever with fully mocked Pinecone."""
        with patch("papermind.rag.retrieval.Pinecone") as MockPC, \
             patch.dict("os.environ", {"PINECONE_API_KEY": "fake", "PINECONE_INDEX": "test"}):
            mock_index = MagicMock()
            MockPC.return_value.Index.return_value = mock_index

            from papermind.rag.retrieval import HybridRetriever
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._index = mock_index
            retriever._bm25 = None
            retriever._corpus = []
            return retriever, mock_index

    def test_dense_search_returns_docs(self):
        retriever, mock_index = self._make_retriever()

        mock_match = MagicMock()
        mock_match.id = "paper1_chunk0"
        mock_match.score = 0.95
        mock_match.metadata = {"text": "transformer attention", "title": "Attention Is All You Need", "arxiv_id": "1706.03762"}
        mock_index.query.return_value.matches = [mock_match]

        with patch("papermind.rag.retrieval._embed_query", return_value=[0.1] * 1536):
            docs = retriever.dense_search("transformer attention", top_k=5)

        assert len(docs) == 1
        assert docs[0]["id"] == "paper1_chunk0"
        assert docs[0]["score"] == pytest.approx(0.95)

    def test_bm25_search_empty_corpus_returns_empty(self):
        retriever, mock_index = self._make_retriever()
        # list() returns empty, so BM25 should gracefully return []
        mock_index.list.return_value = iter([])

        docs = retriever.bm25_search("anything")
        assert docs == []

    def test_search_deduplicates_across_dense_and_bm25(self):
        retriever, mock_index = self._make_retriever()

        shared_doc = {"id": "paper1_chunk0", "text": "shared", "metadata": {}, "score": 0.9}
        unique_bm25 = {"id": "paper2_chunk0", "text": "bm25 only", "metadata": {}, "score": 0.7}

        retriever.dense_search = MagicMock(return_value=[shared_doc])
        retriever.bm25_search = MagicMock(return_value=[shared_doc, unique_bm25])

        with patch("papermind.rag.retrieval.rerank", side_effect=lambda q, docs, top_k: docs[:top_k]):
            from papermind.rag.retrieval import HybridRetriever
            result = retriever.search("query", top_k=5)

        # Should have 2 unique docs (shared_doc deduplicated)
        ids = [d["id"] for d in result]
        assert len(set(ids)) == len(ids)
        assert "paper1_chunk0" in ids
        assert "paper2_chunk0" in ids


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_chunk_text_splits_long_text(self):
        from papermind.rag.ingest import chunk_text

        # Create a long text by repeating words
        long_text = "word " * 2000
        chunks = chunk_text(long_text, arxiv_id="1234.5678", title="Test Paper")

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["arxiv_id"] == "1234.5678"
            assert chunk["title"] == "Test Paper"
            assert "chunk_index" in chunk

    def test_chunk_text_short_text(self):
        from papermind.rag.ingest import chunk_text

        short = "This is a short abstract."
        chunks = chunk_text(short, arxiv_id="0000.0001", title="Short Paper")

        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_indices_are_sequential(self):
        from papermind.rag.ingest import chunk_text

        text = "token " * 3000
        chunks = chunk_text(text, arxiv_id="test", title="T")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
