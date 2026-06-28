#!/usr/bin/env python3
"""
Evaluation harness for PaperMind.

Runs a set of test questions through the pipeline and scores:
  - answer_relevance  (LLM-as-judge, 0–1)
  - hallucination_flag (1 = hallucinated unsupported claim detected)
  - re_retrieval_triggered (1 = critic triggered at least one re-retrieval)

Results written to eval/results.json.

Usage:
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

from anthropic import Anthropic
from papermind.rag.retrieval import HybridRetriever
from papermind.agents.graph import build_graph

EVAL_DIR = Path(__file__).parent.parent / "eval"
RESULTS_FILE = EVAL_DIR / "results.json"

TEST_QUESTIONS = [
    {
        "id": "q1",
        "question": "What are the main advantages of retrieval-augmented generation over fine-tuning?",
        "domain": "RAG",
    },
    {
        "id": "q2",
        "question": "How does the attention mechanism in transformers work?",
        "domain": "Transformers",
    },
    {
        "id": "q3",
        "question": "What methods are used to reduce hallucination in large language models?",
        "domain": "LLMs",
    },
    {
        "id": "q4",
        "question": "How do diffusion models generate images?",
        "domain": "Diffusion",
    },
    {
        "id": "q5",
        "question": "What is reinforcement learning from human feedback and why is it used?",
        "domain": "RLHF",
    },
]


_JUDGE_PROMPT = """\
You are an evaluation judge. Given a research question, a generated answer, and a list of cited paper titles, \
assess the quality of the answer on two dimensions.

1. Relevance score (0.0–1.0): Does the answer address the question accurately and specifically?
   - 1.0 = thorough, accurate, well-grounded answer
   - 0.5 = partially addresses the question or lacks depth
   - 0.0 = off-topic or completely wrong

2. Hallucination flag (0 or 1): Does the answer make specific factual claims (statistics, paper names, \
author names, years) that are NOT supported by the cited papers?
   - 0 = no clear hallucinations detected
   - 1 = at least one suspicious unsupported claim

Respond with JSON only:
{{
  "relevance_score": 0.85,
  "hallucination_flag": 0,
  "reasoning": "one sentence rationale"
}}

Question: {question}

Answer: {answer}

Cited papers: {citations}
"""


def judge_answer(client: Anthropic, question: str, answer: str, citations: list) -> dict:
    citation_titles = "; ".join(c.get("title", "") for c in citations) or "none"
    prompt = _JUDGE_PROMPT.format(question=question, answer=answer, citations=citation_titles)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"relevance_score": 0.5, "hallucination_flag": 0, "reasoning": "parse error"}


def main():
    EVAL_DIR.mkdir(exist_ok=True)

    print("Building pipeline...")
    retriever = HybridRetriever()
    graph = build_graph(retriever=retriever)
    judge_client = Anthropic()

    results = []
    for test in TEST_QUESTIONS:
        q_id = test["id"]
        question = test["question"]
        print(f"\n[{q_id}] {question}")

        t0 = time.time()
        state = graph.invoke({"query": question, "retry_count": 0, "trace": []})
        elapsed = time.time() - t0

        answer = state.get("answer", "")
        citations = state.get("citations", [])
        trace = state.get("trace", [])
        critic_score = state.get("critic_score", None)
        retry_count = state.get("retry_count", 0)

        re_retrieval = 1 if retry_count > 1 else 0

        print(f"  critic_score={critic_score:.2f}  retries={retry_count}  elapsed={elapsed:.1f}s")

        judgment = judge_answer(judge_client, question, answer, citations)
        relevance = judgment.get("relevance_score", 0.5)
        hallucination = judgment.get("hallucination_flag", 0)
        print(f"  relevance={relevance:.2f}  hallucination={hallucination}  — {judgment.get('reasoning', '')}")

        results.append(
            {
                "id": q_id,
                "question": question,
                "domain": test["domain"],
                "answer_snippet": answer[:300],
                "num_citations": len(citations),
                "critic_score": critic_score,
                "retry_count": retry_count,
                "re_retrieval_triggered": re_retrieval,
                "relevance_score": relevance,
                "hallucination_flag": hallucination,
                "judge_reasoning": judgment.get("reasoning", ""),
                "elapsed_s": round(elapsed, 2),
            }
        )

    # Aggregate metrics
    n = len(results)
    summary = {
        "num_questions": n,
        "avg_relevance_score": round(sum(r["relevance_score"] for r in results) / n, 3),
        "hallucination_rate": round(sum(r["hallucination_flag"] for r in results) / n, 3),
        "re_retrieval_rate": round(sum(r["re_retrieval_triggered"] for r in results) / n, 3),
        "avg_critic_score": round(
            sum(r["critic_score"] for r in results if r["critic_score"] is not None) / n, 3
        ),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / n, 2),
    }

    output = {"summary": summary, "results": results}
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n{'='*60}")
    print(f"Evaluation complete — {n} questions")
    print(f"  Avg relevance:      {summary['avg_relevance_score']:.3f}")
    print(f"  Hallucination rate: {summary['hallucination_rate']:.3f}")
    print(f"  Re-retrieval rate:  {summary['re_retrieval_rate']:.3f}")
    print(f"  Avg critic score:   {summary['avg_critic_score']:.3f}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
