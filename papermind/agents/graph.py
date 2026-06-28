"""
LangGraph orchestration for PaperMind.

Flow: Planner → Retriever → Critic → (re-retrieve if needed) → Generator → END
"""
from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from papermind.agents.planner import run_planner
from papermind.agents.retriever import run_retriever
from papermind.agents.critic import run_critic
from papermind.rag.retrieval import HybridRetriever
from papermind.state import ResearchState

MAX_RETRIES = 2
CRITIC_THRESHOLD = 0.6


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)


def _make_planner_node(llm: ChatAnthropic):
    def node(state: ResearchState) -> dict[str, Any]:
        updates = run_planner(state, llm)
        updates["trace"] = state.get("trace", []) + [f"[planner] sub_questions={updates.get('sub_questions')}"]
        return updates
    return node


def _make_retriever_node(retriever: HybridRetriever):
    def node(state: ResearchState) -> dict[str, Any]:
        updates = run_retriever(state, retriever)
        n = len(updates.get("retrieved_docs", []))
        updates["trace"] = state.get("trace", []) + [f"[retriever] fetched {n} docs for query='{state.get('retrieval_query', state['query'])}'"]
        return updates
    return node


def _make_critic_node(llm: ChatAnthropic):
    def node(state: ResearchState) -> dict[str, Any]:
        updates = run_critic(state, llm)
        score = updates.get("critic_score", 0.0)
        updates["trace"] = state.get("trace", []) + [f"[critic] score={score:.2f}"]
        return updates
    return node


def _make_generator_node(llm: ChatAnthropic):
    def node(state: ResearchState) -> dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        query = state["query"]

        context_parts = []
        for i, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            title = meta.get("title", "Unknown")
            arxiv_id = meta.get("arxiv_id", "")
            text = doc.get("text", "")
            context_parts.append(f"[{i+1}] {title} (arXiv:{arxiv_id})\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"You are a research assistant. Answer the following question using ONLY the provided papers.\n"
            f"For every claim, cite the paper number in brackets, e.g. [1].\n"
            f"If the papers don't contain enough information, say so explicitly.\n\n"
            f"Question: {query}\n\n"
            f"Papers:\n{context}\n\n"
            f"Answer:"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content

        citations = [
            {
                "index": i + 1,
                "arxiv_id": doc.get("metadata", {}).get("arxiv_id", ""),
                "title": doc.get("metadata", {}).get("title", ""),
                "authors": doc.get("metadata", {}).get("authors", ""),
            }
            for i, doc in enumerate(docs)
        ]

        trace = state.get("trace", []) + [f"[generator] answer generated ({len(answer)} chars)"]
        return {"answer": answer, "citations": citations, "trace": trace}

    return node


def _critic_router(state: ResearchState) -> str:
    score = state.get("critic_score", 0.0)
    retries = state.get("retry_count", 0)
    if score >= CRITIC_THRESHOLD or retries >= MAX_RETRIES:
        return "generator"
    return "retriever"


def build_graph(retriever: HybridRetriever | None = None) -> Any:
    """Build and compile the LangGraph research pipeline."""
    if retriever is None:
        retriever = HybridRetriever()

    llm = _build_llm()

    graph = StateGraph(ResearchState)
    graph.add_node("planner", _make_planner_node(llm))
    graph.add_node("retriever", _make_retriever_node(retriever))
    graph.add_node("critic", _make_critic_node(llm))
    graph.add_node("generator", _make_generator_node(llm))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_conditional_edges("critic", _critic_router, {"generator": "generator", "retriever": "retriever"})
    graph.add_edge("generator", END)

    return graph.compile()
