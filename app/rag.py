"""Step 2 of RAG: rewrite, retrieve, and answer with citations."""
from functools import lru_cache
import time

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app import config
from app.retrieval import search

NOT_FOUND = "I couldn't find this in the college documents I have."

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for students of a college. Answer the question using ONLY the "
     "context below. If the context does not contain the answer, reply exactly: "
     f"\"{NOT_FOUND}\" Do not guess or use outside knowledge. Keep the answer short and clear."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the latest student question as one standalone question. "
                "Use the conversation only to resolve references such as 'that company' or 'what about it'. "
                "Return only the rewritten question, with no explanation."),
    ("human", "Conversation:\n{history}\n\nLatest question: {question}"),
])


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings   # imported lazily (loads torch)
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(persist_directory=str(config.CHROMA_DIR), embedding_function=get_embeddings())


@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI   # reads GOOGLE_API_KEY from env
    return ChatGoogleGenerativeAI(model=config.LLM_MODEL, timeout=30, max_retries=3)


def format_context(docs):
    return "\n\n".join(f"[{d.metadata.get('source')} p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                        for d in docs)


def rewrite_question(question, history, llm):
    """Resolve a follow-up question before retrieval; leave new questions unchanged."""
    if not history:
        return question
    transcript = "\n".join(
        f"{item.get('role', 'user')}: {item.get('text', '')}" for item in history[-6:]
    )
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"history": transcript, "question": question}).strip()
    return rewritten or question


def _sources_for(docs, text):
    sources, seen = [], set()
    if NOT_FOUND not in text:
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("page", 0) + 1)
            if key not in seen:
                seen.add(key)
                sources.append({"file": key[0], "page": key[1]})
    return sources


def _prepare(question, vectorstore, llm, k, mode, history):
    """Do the shared rewrite/retrieval work for normal and streaming answers."""
    model = llm or get_llm()
    searched_question = rewrite_question(question, history, model)
    retrieval_started = time.perf_counter()
    docs = search(vectorstore or get_vectorstore(), searched_question, k, mode or config.RETRIEVAL_MODE)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    return model, searched_question, docs, retrieval_ms


def answer(question, vectorstore=None, llm=None, k=config.TOP_K, mode=None, history=None):
    """Return answer, sources, rewritten question, and measured retrieval/generation times."""
    model, searched_question, docs, retrieval_ms = _prepare(
        question, vectorstore, llm, k, mode, history or []
    )
    generation_started = time.perf_counter()
    chain = PROMPT | model | StrOutputParser()  # the generation step
    text = chain.invoke({"context": format_context(docs), "question": searched_question})
    generation_ms = (time.perf_counter() - generation_started) * 1000
    text = text.strip()
    return {
        "answer": text,
        "sources": _sources_for(docs, text),
        "searched_question": searched_question,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "latency_ms": {"retrieval": round(retrieval_ms, 1), "generation": round(generation_ms, 1)},
    }


def stream_answer(question, vectorstore=None, llm=None, k=config.TOP_K, mode=None, history=None):
    """Yield answer text chunks; the generator return value contains sources and timings."""
    model, searched_question, docs, retrieval_ms = _prepare(
        question, vectorstore, llm, k, mode, history or []
    )
    generation_started = time.perf_counter()
    chain = PROMPT | model | StrOutputParser()
    parts = []
    for chunk in chain.stream({"context": format_context(docs), "question": searched_question}):
        parts.append(chunk)
        yield chunk
    text = "".join(parts).strip()
    generation_ms = (time.perf_counter() - generation_started) * 1000
    return {
        "answer": text,
        "sources": _sources_for(docs, text),
        "searched_question": searched_question,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "latency_ms": {"retrieval": round(retrieval_ms, 1), "generation": round(generation_ms, 1)},
    }
