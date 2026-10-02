"""Retrieval strategies.  mode = "dense" | "bm25" | "hybrid"

dense  : vector similarity (meaning).         Good at paraphrases, weak at exact names/numbers.
bm25   : keyword scoring (exact words).       Good at rare words like "Diwali", weak at paraphrases.
hybrid : run both, merge with Reciprocal Rank Fusion (RRF): each chunk earns 1/(60+rank) from every
         list it appears in, so chunks that rank well in BOTH lists float to the top.
"""
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())     # keeps decimals like 45.68


@lru_cache(maxsize=2)
def _bm25_index(vs):
    data = vs.get()                                                  # every chunk stored in Chroma
    docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(data["documents"], data["metadatas"])]
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


def bm25_search(vs, query, k):
    bm25, docs = _bm25_index(vs)
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in order if scores[i] > 0]


def hybrid_search(vs, query, k, fetch=20, rrf_k=60):
    rankings = [vs.similarity_search(query, k=fetch), bm25_search(vs, query, fetch)]
    score, keep = {}, {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, 1):
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content)
            score[key] = score.get(key, 0) + 1 / (rrf_k + rank)
            keep[key] = d
    best = sorted(score, key=score.get, reverse=True)[:k]
    return [keep[key] for key in best]


def search(vs, query, k, mode="dense"):
    if mode == "dense":
        return vs.similarity_search(query, k=k)
    if mode == "bm25":
        return bm25_search(vs, query, k)
    if mode == "hybrid":
        return hybrid_search(vs, query, k)
    raise ValueError(f"unknown retrieval mode: {mode}")
