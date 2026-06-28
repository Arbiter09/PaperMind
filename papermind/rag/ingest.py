"""
Ingestion utilities: fetch arXiv papers, chunk text, generate embeddings,
upsert to Pinecone.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import arxiv
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100


def get_pinecone_index(index_name: str | None = None):
    api_key = os.environ["PINECONE_API_KEY"]
    env = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws")
    index_name = index_name or os.environ["PINECONE_INDEX"]

    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        cloud, region = _parse_environment(env)
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)


def _parse_environment(env: str) -> tuple[str, str]:
    """Convert e.g. 'us-east-1-aws' → ('aws', 'us-east-1')."""
    if env.endswith("-aws"):
        return "aws", env[: -len("-aws")]
    if env.endswith("-gcp"):
        return "gcp", env[: -len("-gcp")]
    if env.endswith("-azure"):
        return "azure", env[: -len("-azure")]
    return "aws", env


def fetch_arxiv_papers(query: str, max_results: int = 50) -> list[dict[str, Any]]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        papers.append(
            {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors[:5]),
                "abstract": result.summary,
                "published": str(result.published.date()),
            }
        )
    return papers


def chunk_text(text: str, arxiv_id: str, title: str) -> list[dict[str, Any]]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(
            {
                "text": chunk_text,
                "arxiv_id": arxiv_id,
                "title": title,
                "chunk_index": idx,
            }
        )
        idx += 1
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OpenAI()
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def upsert_to_pinecone(
    index,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    paper_meta: dict[str, Any],
) -> int:
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        vector_id = f"{chunk['arxiv_id']}_chunk{chunk['chunk_index']}"
        metadata = {
            "text": chunk["text"][:1000],
            "arxiv_id": chunk["arxiv_id"],
            "title": chunk["title"],
            "authors": paper_meta.get("authors", ""),
            "abstract": paper_meta.get("abstract", "")[:500],
            "chunk_index": chunk["chunk_index"],
        }
        vectors.append({"id": vector_id, "values": emb, "metadata": metadata})

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    return len(vectors)
