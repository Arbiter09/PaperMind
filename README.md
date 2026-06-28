# PaperMind

A multi-agent research assistant that answers questions about machine learning papers. You ask a question; a small LangGraph pipeline retrieves relevant arXiv papers from a Pinecone vector store, a critic agent decides whether the evidence is good enough, and Claude writes a grounded answer with citations.

Built as a side project to explore multi-agent RAG patterns — not production software, but a real working system.

---

## What it does

1. **Planner** breaks your question into sub-questions and produces a focused retrieval query.
2. **Retriever** runs hybrid search (dense Pinecone embeddings + BM25) and reranks results with a cross-encoder.
3. **Critic** scores the retrieved context against your question. If relevance < 0.6, it reformulates the query and triggers a second retrieval attempt (max 2 retries).
4. **Generator** (Claude Sonnet 4.6) writes an answer grounded in the retrieved papers, with bracketed citations.

The whole pipeline is exposed as a FastAPI server (`POST /query`).

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐
│   Planner   │  breaks query into sub-questions,
│   (Claude)  │  produces retrieval_query
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│              Retriever                      │
│  ┌──────────────┐    ┌─────────────────┐    │
│  │ Dense search │    │   BM25 search   │    │
│  │  (Pinecone)  │    │  (in-memory)    │    │
│  └──────┬───────┘    └────────┬────────┘    │
│         └──────────┬──────────┘             │
│              ┌─────▼──────┐                 │
│              │  Reranker  │  cross-encoder  │
│              └─────┬──────┘                 │
└────────────────────┼────────────────────────┘
                     │ top-6 docs
                     ▼
              ┌──────────────┐
              │    Critic    │  score 0–1
              │   (Claude)   │◄──────────────┐
              └──────┬───────┘               │
                     │                       │ score < 0.6
              score ≥ 0.6               reformulated query
              or retries ≥ 2                 │
                     │                       │
                     └───────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Generator   │  answer + citations
              │   (Claude)   │
              └──────┬───────┘
                     │
                     ▼
             JSON response
```

**Stack:** Python 3.11+ · LangGraph · LangChain · Claude Sonnet 4.6 · OpenAI embeddings · Pinecone · FastAPI · BM25 (rank-bm25) · sentence-transformers

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Arbiter09/PaperMind.git
cd PaperMind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Fill in your keys:
#   ANTHROPIC_API_KEY   — claude.ai or Anthropic console
#   OPENAI_API_KEY      — for text-embedding-3-small
#   PINECONE_API_KEY    — app.pinecone.io
#   PINECONE_INDEX      — e.g. "papermind"
#   PINECONE_ENVIRONMENT — e.g. "us-east-1-aws"
```

### 3. Create the Pinecone index

The ingestion script creates the index automatically if it doesn't exist (1536-dim cosine, serverless). You just need the API key and environment set.

### 4. Ingest papers

```bash
# Default: 5 CS topics, 30 papers each (~150 papers total)
python scripts/ingest.py

# Custom query and paper count
python scripts/ingest.py --query "diffusion models" --max-results 50
```

Ingestion takes 5–15 minutes depending on arXiv rate limits and OpenAI embedding throughput.

### 5. Run the server

```bash
uvicorn papermind.api.main:app --reload --port 8000
```

### 6. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do diffusion models generate images?"}'
```

Response shape:
```json
{
  "answer": "Diffusion models generate images through ...",
  "citations": [
    {"index": 1, "arxiv_id": "2006.11239", "title": "Denoising Diffusion Probabilistic Models", "authors": "Ho et al."}
  ],
  "trace": [
    "[planner] sub_questions=['How does forward diffusion work?', ...]",
    "[retriever] fetched 6 docs for query='diffusion model denoising image generation'",
    "[critic] score=0.85",
    "[generator] answer generated (842 chars)"
  ],
  "critic_score": 0.85
}
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests are fully unit-tested with mocked Pinecone and LLM calls — no API keys needed.

---

## Running the evaluation

```bash
python scripts/evaluate.py
```

Runs 5 test questions through the live pipeline, judges each answer with Claude as a 0–1 relevance scorer, and writes results to `eval/results.json`.

---

## Eval results

Results from a run over ~150 ingested arXiv CS papers (5 queries × 30 papers, mixed topics):

| Metric | Score |
|---|---|
| Avg answer relevance (LLM-judge, 0–1) | **0.820** |
| Hallucination rate | **0.200** |
| Re-retrieval trigger rate | **0.400** |
| Avg critic score | **0.740** |
| Avg end-to-end latency | **8.63 s** |

**Notes:**
- The hallucination flag fires on 1 of 5 questions (the "reducing hallucination" question, ironically — the model cited a specific accuracy number not verifiable in the retrieved chunks).
- Re-retrieval triggered on 2 of 5 questions: the RLHF question had weak initial coverage; the hallucination question got reformulated but still produced a partially unsupported claim.
- Relevance drops on the RLHF question (0.65) because the answer stays high-level — the ingested corpus has fewer RLHF-specific papers relative to transformer architecture papers.

Full per-question breakdown in [`eval/results.json`](eval/results.json).

---

## Project layout

```
papermind/
  agents/
    planner.py    — query decomposition + retrieval query planning
    retriever.py  — calls HybridRetriever, returns top-k docs
    critic.py     — relevance scoring + query reformulation
    graph.py      — LangGraph StateGraph, nodes, conditional edges, generator
  rag/
    ingest.py     — arXiv fetch, chunking, embedding, Pinecone upsert
    retrieval.py  — HybridRetriever (dense + BM25 + rerank)
    reranker.py   — cross-encoder wrapper (ms-marco-MiniLM-L-6-v2)
  api/
    main.py       — FastAPI app, POST /query, GET /health
    schemas.py    — Pydantic request/response models
  state.py        — ResearchState TypedDict (shared across agents)
scripts/
  ingest.py       — CLI for running ingestion
  evaluate.py     — evaluation harness with LLM-as-judge
eval/
  results.json    — latest eval run output
tests/
  test_agents.py  — planner, critic, retriever node unit tests
  test_retrieval.py — reranker, HybridRetriever, chunking tests
```

---

## Limitations / known issues

- BM25 index is built by fetching vector IDs from Pinecone on startup. For large indexes (>10k vectors), this is slow — a local FAISS or SQLite store would be better.
- Chunking is abstract+title only (not full paper PDFs) because the arXiv API doesn't give full text. The `arxiv` library can download PDFs; left as an exercise.
- No streaming — the `/query` endpoint blocks until the full pipeline completes (~8–15s).
- Critic threshold (0.6) and max retries (2) are hardcoded constants in `graph.py`.
