# CLAUDE.md

## What this project is
RAG helpdesk for MKSSS's Cummins College of Engineering for Women. Answers questions from three public PDFs in `data/` (academic calendar, brochure, placement statistics) and always cites file and page. If the documents don't contain the answer it must say so, never guess.

Owner is a fresher building this for off-campus job applications. She wants to **understand every part** and be able to explain it in interviews, so prefer clear, small, commented code over clever abstractions, and explain *why* when changing something.

## Stack
Python 3.12 · LangChain (loaders, splitter, Chroma wrapper, prompt, LLM wrapper) · ChromaDB · sentence-transformers MiniLM (local embeddings) · rank-bm25 · Gemini (via langchain-google-genai) · FastAPI · Streamlit · Docker (deployed as a Hugging Face Space).

## Layout
- `app/config.py`      all settings, overridable by env vars / `.env`
- `app/ingest.py`      PDFs → chunks → embeddings → `chroma_db/` (rebuilds from scratch)
- `app/retrieval.py`   `search(vs, query, k, mode)` with modes dense | bm25 | hybrid (RRF). Hybrid is the default.
- `app/rag.py`         `answer(question)` → `{answer, sources}`. Prompt forces "answer only from context".
- `app/api.py`         FastAPI: `POST /ask`, `GET /health`. Loads the vector store at startup.
- `ui.py`              Streamlit chat UI, calls the API (`API_URL`, default localhost:8000).
- `eval.py`, `questions.json`  retrieval-vs-generation evaluation.

## Commands (venv must be active: `source venv/bin/activate`)
- `python -m app.ingest [--chunk-size N --chunk-overlap M]`  rebuild the index (REQUIRED after changing chunk settings or the PDFs)
- `uvicorn app.api:app --reload`   then `streamlit run ui.py`
- `python eval.py --no-llm --mode hybrid --k 4`  free retrieval check, no API calls
- `python eval.py`  full run with Gemini (about a minute; rate-limited by `time.sleep`)

## Rules
- Never commit `.env`, `chroma_db/`, `venv/`, `*.zip`, `.DS_Store`. Check `git status` before every commit.
- Only public documents go in `data/`. Nothing confidential, and never any employer (e.g. Adobe) material.
- Any change to retrieval, chunking or the prompt must be checked with `eval.py` and the before/after numbers recorded in README.md.
- Never claim accuracy beyond what the 12-question test supports; keep the limitations section honest.
- Keep the "answer only from context / say you couldn't find it" behaviour. Out-of-scope questions must be refused.
- If SQL or other tools are added later: read-only, SELECT only, row limits, schema-only prompts.

## Known gotchas
- Gemini model names change; set `LLM_MODEL` in `.env`. The free tier sometimes returns 503 "high demand", which is temporary (retries are set in `get_llm`).
- `langchain-community` is being phased out (deprecation warning in ingest). It still works; migrating loaders to standalone packages is a possible cleanup.
- The BM25 index is built lazily from Chroma and cached per process. Restart the API after re-ingesting.
- The Hugging Face container runs as user id 1000; keep the non-root user in the Dockerfile.

## Roadmap
1. Deploy (Hugging Face Space, `GOOGLE_API_KEY` as a secret) and put the link in README and resume.
2. LangGraph router: documents vs. placement-data questions, chat memory with query rewriting.
3. Text-to-SQL on the placement table with guardrails.
4. Streaming answers, feedback log, UI polish.
