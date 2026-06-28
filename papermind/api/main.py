"""
PaperMind FastAPI server.

POST /query  — run the full multi-agent pipeline
GET  /health — liveness check
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from papermind.api.schemas import HealthResponse, QueryRequest, QueryResponse, Citation

load_dotenv()

_graph = None
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _retriever
    from papermind.rag.retrieval import HybridRetriever
    from papermind.agents.graph import build_graph

    _retriever = HybridRetriever()
    _graph = build_graph(retriever=_retriever)
    yield


app = FastAPI(
    title="PaperMind",
    description="Multi-agent arXiv research assistant powered by LangGraph and Claude.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse, tags=["research"])
async def query(request: QueryRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    initial_state = {
        "query": request.question,
        "retry_count": 0,
        "trace": [],
    }

    try:
        result = _graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    citations = [
        Citation(
            index=c.get("index", i + 1),
            arxiv_id=c.get("arxiv_id", ""),
            title=c.get("title", ""),
            authors=c.get("authors", ""),
        )
        for i, c in enumerate(result.get("citations", []))
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=citations,
        trace=result.get("trace", []),
        critic_score=result.get("critic_score"),
    )
