"""Evaluation that separates RETRIEVAL from GENERATION.

  python eval.py                 retrieval + Gemini answers (uses the API)
  python eval.py --no-llm        retrieval only: free, fast, no API limits
  python eval.py --k 6           try a different number of retrieved chunks
  python eval.py --mode hybrid   dense | bm25 | hybrid retrieval

Each question in questions.json has:
  question   what the user asks
  expect     text that must appear in the final ANSWER   (or "NOT_FOUND" if the docs can't answer)
  evidence   text that must appear in a RETRIEVED CHUNK  (the passage that really contains the answer)
  file       part of the PDF name expected in the sources
"""
import argparse
import json
import re
import time

from app import config, rag
from app.retrieval import search


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def run(path, k, use_llm, mode):
    questions = json.load(open(path))
    vs = rag.get_vectorstore()
    n_ret = hits = 0
    rr_sum = 0.0
    n_ans = ans_ok = 0
    labels = {}

    for i, q in enumerate(questions, 1):
        refuse_expected = q["expect"] == "NOT_FOUND"
        evidence = None if refuse_expected else q.get("evidence", q["expect"])

        # ---- 1) RETRIEVAL: did the right passage come back? (no LLM involved) ----
        docs = search(vs, q["question"], k, mode)
        rank = None
        if evidence:
            n_ret += 1
            for r, d in enumerate(docs, 1):
                if norm(evidence) in norm(d.page_content):
                    rank = r
                    break
            if rank:
                hits += 1
                rr_sum += 1 / rank

        # ---- 2) GENERATION: did the LLM answer correctly from what it was given? ----
        answered = None
        if use_llm:
            try:
                out = rag.answer(q["question"], k=k, mode=mode)
                text = out["answer"].lower()
                answered = (rag.NOT_FOUND.lower() in text) if refuse_expected else (norm(q["expect"]) in norm(text))
                n_ans += 1
                ans_ok += answered
            except Exception as e:
                print(f"{i:>2}. ERROR      {q['question']} -> {type(e).__name__}")
                continue
            time.sleep(1.5)

        # ---- 3) DIAGNOSIS ----
        if refuse_expected:
            label = "-" if answered is None else ("PASS" if answered else "HALLUCINATION")
        elif answered is None:
            label = "HIT" if rank else "MISS"
        elif rank and answered:
            label = "PASS"
        elif rank and not answered:
            label = "GENERATION"       # right chunk was retrieved, answer still wrong
        elif not rank and answered:
            label = "LUCKY"            # answer right, but evidence not in top-k
        else:
            label = "RETRIEVAL"        # right chunk never reached the LLM
        labels[label] = labels.get(label, 0) + 1

        print(f"{i:>2}. {label:<12} rank={rank if rank else '-':<2} {q['question']}")
        if label in ("MISS", "RETRIEVAL"):
            for d in docs[:2]:
                print(f"       got: {d.metadata.get('source')} p.{d.metadata.get('page', 0) + 1}: "
                      f"{norm(d.page_content)[:90]!r}")

    n = len(questions)
    print(f"\nRetrieval hit rate (k={k}, {mode}): {hits}/{n_ret} = {100 * hits / max(n_ret, 1):.0f}%   "
          f"MRR: {rr_sum / max(n_ret, 1):.2f}")
    if use_llm:
        print(f"Answer accuracy: {ans_ok}/{n_ans} = {100 * ans_ok / max(n_ans, 1):.0f}%")
    print("Breakdown:", ", ".join(f"{v} {k_}" for k_, v in sorted(labels.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--file", default="questions.json")
    a = ap.parse_args()
    run(a.file, a.k, not a.no_llm, a.mode)
