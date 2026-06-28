"""Unit tests for PaperMind agents (planner and critic)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from papermind.agents.planner import run_planner
from papermind.agents.critic import run_critic
from papermind.agents.retriever import run_retriever
from papermind.state import ResearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


def _make_llm(response_content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(response_content)
    return llm


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_returns_sub_questions_and_retrieval_query(self):
        payload = json.dumps({
            "sub_questions": ["What is RAG?", "How does RAG reduce hallucination?"],
            "retrieval_query": "retrieval augmented generation hallucination",
        })
        llm = _make_llm(payload)
        state: ResearchState = {"query": "How does RAG reduce hallucination?"}

        result = run_planner(state, llm)

        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 2
        assert result["retrieval_query"] == "retrieval augmented generation hallucination"
        assert result["retry_count"] == 0

    def test_falls_back_on_bad_json(self):
        llm = _make_llm("not valid json at all")
        state: ResearchState = {"query": "What is attention?"}

        result = run_planner(state, llm)

        assert result["sub_questions"] == ["What is attention?"]
        assert result["retrieval_query"] == "What is attention?"

    def test_strips_markdown_fences(self):
        payload = "```json\n" + json.dumps({
            "sub_questions": ["X?"],
            "retrieval_query": "X mechanism",
        }) + "\n```"
        llm = _make_llm(payload)
        state: ResearchState = {"query": "X?"}

        result = run_planner(state, llm)
        assert result["retrieval_query"] == "X mechanism"

    def test_single_focused_question_passes_through(self):
        payload = json.dumps({
            "sub_questions": ["What is BERT?"],
            "retrieval_query": "BERT language model",
        })
        llm = _make_llm(payload)
        state: ResearchState = {"query": "What is BERT?"}

        result = run_planner(state, llm)
        assert result["sub_questions"] == ["What is BERT?"]


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class TestCritic:
    def _make_state(self, score_payload: dict, retry_count: int = 0) -> ResearchState:
        docs = [
            {"text": "This paper discusses RAG.", "metadata": {"title": "RAG paper", "arxiv_id": "1234.5678"}}
        ]
        return {
            "query": "What is RAG?",
            "retrieved_docs": docs,
            "retry_count": retry_count,
        }

    def test_high_score_passes_through(self):
        payload = json.dumps({"score": 0.9, "reformulated_query": None})
        llm = _make_llm(payload)
        state = self._make_state(payload)

        result = run_critic(state, llm)

        assert result["critic_score"] == pytest.approx(0.9)
        assert "retrieval_query" not in result

    def test_low_score_sets_reformulated_query(self):
        payload = json.dumps({"score": 0.3, "reformulated_query": "better search terms"})
        llm = _make_llm(payload)
        state = self._make_state(payload)

        result = run_critic(state, llm)

        assert result["critic_score"] == pytest.approx(0.3)
        assert result.get("retrieval_query") == "better search terms"

    def test_increments_retry_count(self):
        payload = json.dumps({"score": 0.5, "reformulated_query": "new query"})
        llm = _make_llm(payload)
        state = self._make_state(payload, retry_count=1)

        result = run_critic(state, llm)
        assert result["retry_count"] == 2

    def test_fallback_on_bad_json(self):
        llm = _make_llm("oops not json")
        state = self._make_state({})

        result = run_critic(state, llm)
        assert 0.0 <= result["critic_score"] <= 1.0

    def test_no_reformulation_above_threshold(self):
        payload = json.dumps({"score": 0.8, "reformulated_query": "ignored"})
        llm = _make_llm(payload)
        state = self._make_state(payload)

        result = run_critic(state, llm)
        # Score is >= 0.6, so reformulated_query should NOT be set
        assert "retrieval_query" not in result


# ---------------------------------------------------------------------------
# Retriever node
# ---------------------------------------------------------------------------

class TestRetrieverNode:
    def test_uses_retrieval_query_from_state(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [{"text": "paper text", "metadata": {}}]

        state: ResearchState = {
            "query": "original question",
            "retrieval_query": "refined search terms",
        }

        result = run_retriever(state, mock_retriever)

        mock_retriever.search.assert_called_once_with("refined search terms", top_k=6)
        assert len(result["retrieved_docs"]) == 1

    def test_falls_back_to_query_when_no_retrieval_query(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        state: ResearchState = {"query": "fallback query"}
        run_retriever(state, mock_retriever)

        mock_retriever.search.assert_called_once_with("fallback query", top_k=6)
