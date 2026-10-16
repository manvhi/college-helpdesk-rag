"""Streamlit chat UI.

Set API_URL to use the FastAPI service. Without it, the app runs the RAG pipeline
inside Streamlit, which is the deployment mode used by Streamlit Community Cloud.
"""
import json
import os

import requests
import streamlit as st


# Streamlit Community Cloud stores secrets in st.secrets, not necessarily in the environment.
if not os.getenv("GOOGLE_API_KEY"):
    try:
        secret_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secret_key = None
    if secret_key:
        os.environ["GOOGLE_API_KEY"] = secret_key

API_URL = os.getenv("API_URL", "").strip().rstrip("/")


@st.cache_resource
def prepare_local_index():
    """Build the local index once if it is absent, then return the vector store."""
    from app import config, rag

    if not config.CHROMA_DIR.exists():
        from app.ingest import build_index
        build_index()
        rag.get_vectorstore.cache_clear()
    return rag.get_vectorstore()


def local_answer(question, history):
    from app import rag

    prepare_local_index()
    return rag.answer(question, history=history)


def stream_api_answer(question, history, metadata):
    """Yield API text chunks and collect the final SSE metadata in a dictionary."""
    response = requests.post(
        f"{API_URL}/ask/stream",
        json={"question": question, "history": history},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    event = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            event = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if event == "done":
            metadata.update(payload)
            return
        if event == "error":
            raise RuntimeError(payload.get("detail", "Streaming request failed"))
        if payload.get("text"):
            yield payload["text"]
        event = ""


st.set_page_config(page_title="College Helpdesk", page_icon="🎓")
st.title("🎓 College Student Helpdesk")
st.caption("Ask about the college documents. Answers include file and page citations.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg.get("searched_question") and msg.get("question") != msg["searched_question"]:
            st.caption(f"Searched for: {msg['searched_question']}")
        if msg.get("latency_ms"):
            st.caption(f"Latency: {msg['latency_ms'].get('total', 0):.0f} ms")
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(f"📄 {source['file']} — page {source['page']}")

if question := st.chat_input("Ask a question..."):
    # Copy the previous turns before adding this question: the new question is not its own context.
    prior_history = st.session_state.history[-6:]
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            if API_URL:
                data = {}
                answer_text = st.write_stream(stream_api_answer(question, prior_history, data))
                data["answer"] = answer_text
                already_rendered = True
            else:
                data = local_answer(question, prior_history)
                already_rendered = False

            if not already_rendered:
                st.write(data["answer"])
            if data.get("searched_question") and data["searched_question"] != question:
                st.caption(f"Searched for: {data['searched_question']}")
            if data.get("latency_ms"):
                st.caption(f"Latency: {data['latency_ms'].get('total', 0):.0f} ms")
            if data.get("sources"):
                with st.expander("Sources"):
                    for source in data["sources"]:
                        st.write(f"📄 {source['file']} — page {source['page']}")
            st.session_state.history.append({
                "role": "assistant",
                "question": question,
                "text": data["answer"],
                "sources": data.get("sources", []),
                "searched_question": data.get("searched_question", question),
                "latency_ms": data.get("latency_ms", {}),
            })
        except Exception as error:
            st.error(f"Helpdesk error: {error}")
