"""
Critic agent — scores retrieved documents for relevance to the query.
If the score is below the threshold, it reformulates the retrieval query
so the retriever can try again.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from papermind.state import ResearchState


_CRITIC_PROMPT = """\
You are a relevance critic. You will be given a research question and a set of retrieved paper abstracts/chunks.

Your job:
1. Score the overall relevance of the retrieved documents to the question on a scale of 0.0 to 1.0.
   - 1.0 = all documents are directly relevant and sufficient to answer the question
   - 0.5 = some documents are relevant but coverage is partial
   - 0.0 = documents are unrelated to the question
2. If the score is below 0.6, provide a better retrieval query (≤15 words) that might find more relevant papers.

Respond with JSON only. Example:
{{
  "score": 0.75,
  "reformulated_query": null
}}

Or if reformulation is needed:
{{
  "score": 0.4,
  "reformulated_query": "transformer attention mechanism efficient inference"
}}

Research question: {query}

Retrieved documents:
{docs_text}
"""


def _format_docs(docs: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(docs[:6]):
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        text = doc.get("text", "")[:400]
        parts.append(f"[{i+1}] {title}\n{text}")
    return "\n\n".join(parts) if parts else "(no documents retrieved)"


def run_critic(state: ResearchState, llm: ChatAnthropic) -> dict[str, Any]:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    docs_text = _format_docs(docs)
    prompt = _CRITIC_PROMPT.format(query=query, docs_text=docs_text)

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        reformulated = parsed.get("reformulated_query")
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        reformulated = None

    updates: dict[str, Any] = {
        "critic_score": score,
        "retry_count": retry_count + 1,
    }

    if reformulated and score < 0.6:
        updates["retrieval_query"] = reformulated

    return updates
