"""Step 2 of RAG: retrieve relevant chunks for a question -> give them to the LLM -> answer with sources."""
from functools import lru_cache

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
    return ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0, timeout=30, max_retries=3)


def format_context(docs):
    return "\n\n".join(f"[{d.metadata.get('source')} p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                       for d in docs)


def answer(question, vectorstore=None, llm=None, k=config.TOP_K, mode=None):
    """Returns {"answer": str, "sources": [{"file":..., "page":...}, ...]}"""
    vs = vectorstore or get_vectorstore()
    docs = search(vs, question, k, mode or config.RETRIEVAL_MODE)   # <- the "retrieval" step
    chain = PROMPT | (llm or get_llm()) | StrOutputParser()          # <- the "generation" step
    text = chain.invoke({"context": format_context(docs), "question": question})

    sources, seen = [], set()
    if NOT_FOUND not in text:                                  # don't cite sources for "not found"
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("page", 0) + 1)
            if key not in seen:
                seen.add(key)
                sources.append({"file": key[0], "page": key[1]})
    return {"answer": text.strip(), "sources": sources}
