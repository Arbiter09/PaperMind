"""Pydantic request/response schemas for the PaperMind API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The research question to answer.")


class Citation(BaseModel):
    index: int
    arxiv_id: str
    title: str
    authors: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[str]
    critic_score: float | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
