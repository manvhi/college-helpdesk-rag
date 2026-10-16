"""FastAPI backend. Run with: uvicorn app.api:app --reload"""
from contextlib import asynccontextmanager
import json
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from app import rag
from app import retrieval

@asynccontextmanager
async def lifespan(app):
    rag.get_vectorstore()          # load the embedding model + Chroma once at startup, not on the first question
    retrieval._get_reranker()      # download/load the cross-encoder before the first request
    yield


app = FastAPI(title="College Student Helpdesk (RAG)", lifespan=lifespan)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=2000)

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    history: list[Message] = Field(default_factory=list, max_length=10)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Question):
    started = time.perf_counter()
    try:
        result = rag.answer(q.question, history=[m.model_dump() for m in q.history])
        result["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000, 1)
        return result
    except Exception as e:
        raise _request_error(e)


def _request_error(error):
    """Turn common provider outages into a useful response while preserving other errors."""
    message = f"{type(error).__name__}: {error}"
    if "503" in message or "unavailable" in message.lower() or "high demand" in message.lower():
        return HTTPException(status_code=503, detail="Gemini is temporarily unavailable; please try again shortly.")
    return HTTPException(status_code=500, detail=message)


@app.post("/ask/stream")
def ask_stream(q: Question):
    """Stream Server-Sent Events, ending with sources and timing metadata."""
    def events():
        started = time.perf_counter()
        iterator = None
        try:
            iterator = rag.stream_answer(
                q.question, history=[m.model_dump() for m in q.history]
            )
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration as finished:
                    result = finished.value
                    result["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000, 1)
                    yield f"event: done\ndata: {json.dumps(result)}\n\n"
                    return
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as error:
            detail = _request_error(error).detail
            yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
