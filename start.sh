#!/bin/sh
# Runs the API in the background and the UI in the foreground (used by Docker)
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
exec streamlit run ui.py --server.port=${PORT:-7860} --server.address=0.0.0.0
