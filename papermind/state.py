"""Shared LangGraph state definition for PaperMind."""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    query: str
    sub_questions: list[str]
    retrieval_query: str
    retrieved_docs: list[dict[str, Any]]
    critic_score: float
    retry_count: int
    answer: str
    citations: list[dict[str, Any]]
    trace: list[str]
