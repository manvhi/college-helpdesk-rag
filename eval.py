"""Run:  python eval.py            (uses questions.json)
For each question: is the expected text in the answer? Is the expected PDF among the sources?"""
import json
import sys
import time

from app import rag


def run(path="questions.json"):
    questions = json.load(open(path))
    answer_ok = source_ok = source_total = 0
    for i, q in enumerate(questions, 1):
        try:
            out = rag.answer(q["question"])
        except Exception as e:
            print(f"{i:>2}. ERROR  {q['question']}  ->  {type(e).__name__}")
            continue
        text = out["answer"].lower()
        if q["expect"] == "NOT_FOUND":
            a = rag.NOT_FOUND.lower() in text
        else:
            a = q["expect"].lower() in text
        answer_ok += a
        s = ""
        if q.get("file"):
            source_total += 1
            ok = any(q["file"].lower() in src["file"].lower() for src in out["sources"])
            source_ok += ok
            s = "  source:" + ("OK" if ok else "WRONG")
        print(f"{i:>2}. {'PASS' if a else 'FAIL'}  {q['question']}{s}")
        if not a:
            print(f"      expected: {q['expect']!r}\n      got:      {out['answer'][:150]!r}")
        time.sleep(1.5)                      # be gentle with the free API limits
    n = len(questions)
    print(f"\nAnswer accuracy: {answer_ok}/{n} = {100 * answer_ok / n:.0f}%")
    if source_total:
        print(f"Correct source:  {source_ok}/{source_total} = {100 * source_ok / source_total:.0f}%")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "questions.json")
