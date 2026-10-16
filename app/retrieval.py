"""Retrieval strategies.  mode = "dense" | "bm25" | "hybrid" | "rerank"

dense  : vector similarity (meaning).         Good at paraphrases, weak at exact names/numbers.
bm25   : keyword scoring (exact words).       Good at rare words like "Diwali", weak at paraphrases.
hybrid : run both, merge with Reciprocal Rank Fusion (RRF): each chunk earns 1/(60+rank) from every
         list it appears in, so chunks that rank well in BOTH lists float to the top.
rerank : 2-stage pipeline. Stage 1 = hybrid search fetches ~10 candidates (fast, a bit rough).
         Stage 2 = a cross-encoder reads (question, chunk) TOGETHER and re-scores each candidate
         (slow but accurate, so we only run it on the short list). Keep the best k.

Parent-document step (after any mode): the search runs over small CHILD chunks, then each hit is
replaced by its larger PARENT chunk, so the LLM sees the full surrounding text.
"""
import json
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app import config


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


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder      # imported lazily (loads torch); model downloads once
    return CrossEncoder(config.RERANK_MODEL)


def rerank(query, candidates, top_n):
    """Score every (query, chunk) pair with the cross-encoder, return the top_n best chunks."""
    if not candidates:
        return []
    scores = _get_reranker().predict([(query, d.page_content) for d in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def search_children(vs, query, k, mode="dense"):
    """Search the small child chunks with the chosen mode."""
    if mode == "dense":
        return vs.similarity_search(query, k=k)
    if mode == "bm25":
        return bm25_search(vs, query, k)
    if mode == "hybrid":
        return hybrid_search(vs, query, k)
    if mode == "rerank":
        candidates = hybrid_search(vs, query, max(k, config.RERANK_CANDIDATES))   # stage 1: wide net
        return rerank(query, candidates, k)                                       # stage 2: precise
    raise ValueError(f"unknown retrieval mode: {mode}")


@lru_cache(maxsize=1)
def _load_parents():
    """parent_id -> {text, source, page}. Written by app.ingest. Restart the API after re-ingesting."""
    if not config.PARENTS_FILE.exists():
        return {}
    return json.loads(config.PARENTS_FILE.read_text(encoding="utf-8"))


def expand_to_parents(children):
    """Replace each child hit by its parent chunk, keeping the ranking order.
    Two children from the same parent give ONE parent (no duplicated context)."""
    parents = _load_parents()
    if not parents:                                    # index built without parents: use children as they are
        return children
    result, seen = [], set()
    for child in children:
        pid = child.metadata.get("parent_id")
        if pid not in parents:
            result.append(child)
            continue
        if pid in seen:
            continue
        seen.add(pid)
        p = parents[pid]
        result.append(Document(page_content=p["text"],
                               metadata={"source": p["source"], "page": p["page"], "parent_id": pid}))
    return result


def search(vs, query, k, mode="dense"):
    """The function the rest of the app calls: search children, return parents."""
    return expand_to_parents(search_children(vs, query, k, mode))
