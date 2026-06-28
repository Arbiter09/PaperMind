#!/usr/bin/env python3
"""
Fetch arXiv CS papers, chunk them, generate embeddings, and upsert into Pinecone.

Usage:
    python scripts/ingest.py --query "large language models" --max-results 50
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

from papermind.rag.ingest import (
    fetch_arxiv_papers,
    chunk_text,
    embed_texts,
    upsert_to_pinecone,
    get_pinecone_index,
)


DEFAULT_QUERIES = [
    "large language models transformer",
    "retrieval augmented generation RAG",
    "attention mechanism neural network",
    "diffusion models image generation",
    "reinforcement learning from human feedback",
]


def ingest_query(index, query: str, max_results: int, verbose: bool = True) -> int:
    if verbose:
        print(f"\n→ Fetching up to {max_results} papers for: '{query}'")

    papers = fetch_arxiv_papers(query, max_results=max_results)
    if verbose:
        print(f"  fetched {len(papers)} papers")

    total_vectors = 0
    for paper in papers:
        text = f"{paper['title']}\n\n{paper['abstract']}"
        chunks = chunk_text(text, arxiv_id=paper["arxiv_id"], title=paper["title"])
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        n = upsert_to_pinecone(index, chunks, embeddings, paper)
        total_vectors += n
        if verbose:
            print(f"  [{paper['arxiv_id']}] '{paper['title'][:60]}' → {n} vectors")

    return total_vectors


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into Pinecone.")
    parser.add_argument("--query", nargs="+", default=None, help="arXiv search query(ies). Uses defaults if not set.")
    parser.add_argument("--max-results", type=int, default=30, help="Papers per query (default 30).")
    args = parser.parse_args()

    queries = args.query if args.query else DEFAULT_QUERIES

    index = get_pinecone_index()
    print(f"Connected to Pinecone index: {os.environ['PINECONE_INDEX']}")

    grand_total = 0
    for q in queries:
        grand_total += ingest_query(index, q, max_results=args.max_results)

    print(f"\n✓ Ingestion complete. Total vectors upserted: {grand_total}")


if __name__ == "__main__":
    main()
