#!/bin/sh
set -e

# The API is useful for local clients; Streamlit is the public foreground process.
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
export API_URL="http://localhost:8000"
exec streamlit run ui.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
