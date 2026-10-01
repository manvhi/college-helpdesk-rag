"""Simple chat UI.  Run:  streamlit run ui.py   (FastAPI must be running too)"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="College Helpdesk", page_icon="🎓")
st.title("🎓 College Student Helpdesk")
st.caption("Ask about syllabus, exam rules, attendance, calendar... Answers come only from the college documents.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"📄 {s['file']} — page {s['page']}")

if question := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
            r.raise_for_status()
            data = r.json()
            st.write(data["answer"])
            if data["sources"]:
                with st.expander("Sources"):
                    for s in data["sources"]:
                        st.write(f"📄 {s['file']} — page {s['page']}")
            st.session_state.history.append({"role": "assistant", "text": data["answer"], "sources": data["sources"]})
        except Exception as e:
            st.error(f"Backend error: {e}")
