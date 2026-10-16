"""Small, repeatable evaluation for retrieval, generation, and Ragas.

Examples:
  venv/bin/python eval.py --no-llm --mode rerank --k 3
  venv/bin/python eval.py --mode rerank --k 3
  venv/bin/python eval.py --ragas --mode rerank --k 3

Each question has an ``evidence`` phrase for retrieval checks. Questions with
``history`` exercise follow-up rewriting. ``retrieval_question`` is the
standalone equivalent used by the free retrieval-only check, so that the
check does not need an LLM just to rewrite a follow-up.
"""
import argparse
import json
import re
import sys
import time
import types
from pathlib import Path

from app import config, rag
from app.retrieval import search


def norm(value):
    return re.sub(r"\s+", " ", value).lower()


def standalone_question(item):
    return item.get("retrieval_question", item["question"])


def retrieval_stats(questions, vs, k, mode):
    """Return per-question ranks and aggregate Hit@k/MRR without an API call."""
    rows, hits, rr_sum, n_ret = [], 0, 0.0, 0
    for item in questions:
        evidence = None if item["expect"] == "NOT_FOUND" else item.get("evidence", item["expect"])
        docs = search(vs, standalone_question(item), k, mode)
        rank = None
        if evidence:
            n_ret += 1
            for current_rank, doc in enumerate(docs, 1):
                if norm(evidence) in norm(doc.page_content):
                    rank = current_rank
                    break
            if rank:
                hits += 1
                rr_sum += 1 / rank
        rows.append({
            "question": item["question"],
            "rank": rank,
            "retrieved": [
                {"file": d.metadata.get("source"), "page": d.metadata.get("page", 0) + 1}
                for d in docs
            ],
        })
    return rows, hits, rr_sum / max(n_ret, 1), n_ret


def run(path, k, use_llm, mode):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    vs = rag.get_vectorstore()
    rows, hits, mrr, n_ret = retrieval_stats(questions, vs, k, mode)
    labels = {}
    answer_rows = []
    answer_ok = 0
    answer_count = 0

    by_question = {row["question"]: row for row in rows}
    for number, item in enumerate(questions, 1):
        row = by_question[item["question"]]
        rank = row["rank"]
        answered = None
        error_type = None
        if use_llm:
            answer_count += 1
            try:
                result = rag.answer(
                    item["question"],
                    k=k,
                    mode=mode,
                    history=item.get("history"),
                )
                text = norm(result["answer"])
                expected_not_found = item["expect"] == "NOT_FOUND"
                answered = (norm(rag.NOT_FOUND) in text) if expected_not_found else norm(item["expect"]) in text
                answer_ok += int(answered)
                answer_rows.append({
                    "question": item["question"],
                    "answer": result["answer"],
                    "searched_question": result.get("searched_question"),
                    "correct": answered,
                    "latency_ms": result.get("latency_ms"),
                })
            except Exception as error:
                error_type = type(error).__name__
                answer_rows.append({
                    "question": item["question"],
                    "error": error_type,
                    "correct": False,
                })
                print(f"{number:>2}. ERROR      {item['question']} -> {error_type}")
            time.sleep(1.5)  # stay below free-tier request limits

        expected_not_found = item["expect"] == "NOT_FOUND"
        if error_type:
            label = "ERROR"
        elif expected_not_found:
            label = "-" if answered is None else ("PASS" if answered else "HALLUCINATION")
        elif answered is None:
            label = "HIT" if rank else "MISS"
        elif rank and answered:
            label = "PASS"
        elif rank and not answered:
            label = "GENERATION"
        elif not rank and answered:
            label = "LUCKY"
        else:
            label = "RETRIEVAL"
        labels[label] = labels.get(label, 0) + 1
        print(f"{number:>2}. {label:<12} rank={rank if rank else '-':<2} {item['question']}")
        if label in ("MISS", "RETRIEVAL"):
            for doc in search(vs, standalone_question(item), k, mode)[:2]:
                print(f"       got: {doc.metadata.get('source')} p.{doc.metadata.get('page', 0) + 1}: "
                      f"{norm(doc.page_content)[:90]!r}")

    n = len(questions)
    hit_rate = 100 * hits / max(n_ret, 1)
    print(f"\nRetrieval hit rate (k={k}, {mode}): {hits}/{n_ret} = {hit_rate:.0f}%   MRR: {mrr:.2f}")
    if use_llm:
        errors = sum(1 for row in answer_rows if row.get("error"))
        print(f"Answer accuracy: {answer_ok}/{answer_count} = {100 * answer_ok / max(answer_count, 1):.0f}% "
              f"({errors} provider error(s))")
    print("Breakdown:", ", ".join(f"{count} {label}" for label, count in sorted(labels.items())))
    print("\n| Mode | k | Hit@k | MRR | Answer accuracy |")
    print("|---|---:|---:|---:|---:|")
    accuracy = "n/a" if not use_llm else f"{answer_ok}/{answer_count}"
    print(f"| {mode} | {k} | {hits}/{n_ret} ({hit_rate:.0f}%) | {mrr:.2f} | {accuracy} |")
    return {
        "mode": mode,
        "k": k,
        "hit_at_k": hits / max(n_ret, 1),
        "mrr": mrr,
        "answer_accuracy": None if not use_llm else answer_ok / max(answer_count, 1),
        "labels": labels,
        "questions": rows,
        "answers": answer_rows,
    }


def _install_ragas_compatibility():
    """Ragas 0.3 imports a removed optional LangChain Vertex module.

    This tiny placeholder is only used for Ragas' type registry; this project
    evaluates with the Gemini LangChain model below, not Vertex AI.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = module


def run_ragas(path, k, mode):
    """Run three LLM-judge metrics on the first ten answerable references."""
    _install_ragas_compatibility()
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextRecall, Faithfulness

    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = [item for item in questions if item.get("reference") and item["expect"] != "NOT_FOUND"][:10]
    vs = rag.get_vectorstore()
    _retrieval_rows, hits, mrr, n_ret = retrieval_stats(selected, vs, k, mode)
    samples = []
    for item in selected:
        result = rag.answer(item["question"], k=k, mode=mode, history=item.get("history"))
        docs = search(vs, standalone_question(item), k, mode)
        samples.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [doc.page_content for doc in docs],
            "reference": item["reference"],
        })
        time.sleep(1.5)

    dataset = EvaluationDataset.from_list(samples)
    evaluator_llm = LangchainLLMWrapper(rag.get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(rag.get_embeddings())
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    frame = result.to_pandas()
    records = json.loads(frame.to_json(orient="records"))
    metric_names = ["faithfulness", "answer_relevancy", "context_recall"]
    averages = {
        name: float(frame[name].mean()) if name in frame else None for name in metric_names
    }
    payload = {
        "mode": mode,
        "k": k,
        "hit_at_k": hits / max(n_ret, 1),
        "mrr": mrr,
        "questions": records,
        "averages": averages,
    }
    Path("eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nRagas results")
    print("| Faithfulness | Answer relevancy | Context recall |")
    print("|---:|---:|---:|")
    print(f"| {averages['faithfulness']:.3f} | {averages['answer_relevancy']:.3f} | {averages['context_recall']:.3f} |")
    print("Per-question results written to eval_results.json")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE,
                        choices=["dense", "bm25", "hybrid", "rerank"])
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--file", default="questions.json")
    args = parser.parse_args()
    if args.ragas:
        run_ragas(args.file, args.k, args.mode)
    else:
        run(args.file, args.k, not args.no_llm, args.mode)
