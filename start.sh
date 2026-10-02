#!/bin/sh
# API in the background (port 8000), chat UI in the foreground (port 7860 = the public port)
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
exec streamlit run ui.py \
  --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true \
  --server.enableCORS=false --server.enableXsrfProtection=false
