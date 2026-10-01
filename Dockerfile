FROM python:3.11-slim
WORKDIR /app

# CPU-only torch keeps the image much smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Build the vector index at image build time (downloads the embedding model once)
RUN python -m app.ingest

ENV PORT=7860
EXPOSE 7860
# Pass the key at runtime:  docker run -e GOOGLE_API_KEY=... -p 7860:7860 helpdesk
CMD ["./start.sh"]
