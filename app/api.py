"""FastAPI backend.  Run:  uvicorn app.api:app --reload   then open http://localhost:8000/docs"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import rag


@asynccontextmanager
async def lifespan(app):
    rag.get_vectorstore()          # load the embedding model + Chroma once at startup, not on the first question
    yield


app = FastAPI(title="College Student Helpdesk (RAG)", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    try:
        return rag.answer(q.question)
    except Exception as e:                                     # e.g. missing API key / no index yet
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
