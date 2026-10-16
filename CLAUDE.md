# CLAUDE.md

## What this project is

RAG helpdesk for MKSSS's Cummins College of Engineering for Women. It answers from three public PDFs in `data/pdfs/` (academic calendar, brochure, and placement statistics) and cites the source file and page. If the documents do not contain an answer, it must say so and never guess.

The owner is a fresher preparing for off-campus jobs. Prefer small, clear, commented code that she can explain in interviews. Explain why when changing behaviour.

## Stack

Python 3.12 · LangChain · ChromaDB · sentence-transformers MiniLM embeddings · rank-bm25 · cross-encoder reranking · Gemini via `langchain-google-genai` · FastAPI · Streamlit · Docker · optional Ragas evaluation.

## Layout

- `app/config.py` — settings, overridable with environment variables / `.env`; defaults are child chunks 400/60, parent chunks 2000/200, `TOP_K=3`, `RETRIEVAL_MODE="rerank"`, reranker candidates 10.
- `app/ingest.py` — recursively loads `data/`, creates child and parent chunks, rebuilds `chroma_db/` and `parents.json`.
- `app/retrieval.py` — `search(vs, query, k, mode)` with `dense | bm25 | hybrid | rerank`; rerank uses hybrid candidates and then expands hits to parent chunks.
- `app/rag.py` — follow-up rewrite, context-only prompt, `answer()` with sources/timing, and `stream_answer()`.
- `app/api.py` — FastAPI `POST /ask`, `POST /ask/stream`, and `GET /health`; preloads the vector store and cross-encoder.
- `ui.py` — Streamlit UI; calls FastAPI when `API_URL` is set, otherwise runs `rag.answer()` in-process for Community Cloud.
- `eval.py`, `questions.json` — retrieval/generation evaluation, optional history, references, and Ragas; `eval_results.json` is generated output.
- `Dockerfile`, `entrypoint.sh` — Python 3.12 CPU image and API/Streamlit launcher.

## Commands

Activate the venv first: `source venv/bin/activate`.

- `python -m app.ingest` — rebuild the index; required after changing PDFs or chunk settings.
- `uvicorn app.api:app --reload` and `streamlit run ui.py` — local API + UI mode.
- `streamlit run ui.py` — in-process UI mode.
- `python eval.py --no-llm --mode dense --k 3` — free retrieval check.
- `python eval.py --no-llm --mode bm25 --k 3` — free keyword check.
- `python eval.py --no-llm --mode hybrid --k 3` — free hybrid check.
- `python eval.py --no-llm --mode rerank --k 3` — free reranker check.
- `python eval.py --mode rerank --k 3` — generation evaluation with Gemini; sleeps between calls.
- `python eval.py --ragas --mode rerank --k 3` — small Ragas run; writes `eval_results.json`.

## Rules

- Never commit `.env`, `chroma_db/`, `venv/`, `*.zip`, `.DS_Store`, or private credentials.
- Only public PDFs belong in `data/pdfs/`.
- Any retrieval, chunking, prompt, or evaluation change must be checked with `eval.py`, and the measured numbers must be recorded in `README.md`.
- Never claim accuracy beyond the small hand-written set. Keep provider errors and Ragas limitations visible.
- Keep the context-only prompt and exact not-found behaviour: `I couldn't find this in the college documents I have.`
- Keep `answer()` backward compatible; new parameters belong at the end.
- If adding SQL later, make it read-only with SELECT-only prompts, schema-only context, and row limits.
- Run `git status` before every commit. Commit locally in small logical commits and never push unless explicitly asked.

## Known gotchas

- Gemini model names change; set `LLM_MODEL` in `.env`. Some Gemini models ignore sampling parameters and the free tier can return 503/high-demand errors or timeouts.
- `langchain-community` loaders currently work but emit a deprecation warning.
- BM25 and parent metadata are cached per process. Restart the API after re-ingesting.
- The cross-encoder is loaded lazily by retrieval and eagerly by FastAPI lifespan; the first rerank call is much slower than warm calls.
- Retrieval peak memory is about 510 MiB with the current PyTorch/MiniLM/cross-encoder stack; an ONNX/fastembed fallback may be needed for a tight deployment limit.
- Ragas 0.3.9 currently imports an optional removed Vertex module from the installed LangChain community package. `eval.py` provides a small type-registry compatibility shim; Ragas still uses Gemini and local MiniLM here.

## Roadmap

1. Deploy the in-process Streamlit app on Streamlit Community Cloud with `GOOGLE_API_KEY` in Secrets.
2. Add guarded, read-only text-to-SQL for placement aggregations.
3. Add streaming UI polish and feedback logging.
4. Add permission-based college-site sources if the college authorizes them.
