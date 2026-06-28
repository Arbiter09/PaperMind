"""
Planner agent — decomposes the user query into sub-questions and
produces the initial retrieval query.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from papermind.state import ResearchState


_PLANNER_PROMPT = """\
You are a research planning assistant. Given a user research question, do two things:

1. Break the question into 1-3 focused sub-questions that together cover the original question.
   If the question is already focused, one sub-question is fine.
2. Produce a concise retrieval query (≤15 words) optimised for searching an arXiv paper index.

Respond with JSON only — no prose, no markdown fences. Example:
{{
  "sub_questions": ["What is X?", "How does Y relate to X?"],
  "retrieval_query": "X Y relationship mechanism"
}}

User question: {query}
"""


def run_planner(state: ResearchState, llm: ChatAnthropic) -> dict[str, Any]:
    query = state["query"]
    prompt = _PLANNER_PROMPT.format(query=query)

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        parsed = json.loads(raw)
        sub_questions = parsed.get("sub_questions", [query])
        retrieval_query = parsed.get("retrieval_query", query)
    except json.JSONDecodeError:
        sub_questions = [query]
        retrieval_query = query

    return {
        "sub_questions": sub_questions,
        "retrieval_query": retrieval_query,
        "retry_count": 0,
    }
