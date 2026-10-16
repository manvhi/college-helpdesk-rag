FROM python:3.12-slim

# Run the application as a non-root user.
RUN useradd --create-home --uid 1000 appuser
USER appuser
WORKDIR /home/appuser/app

ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    HF_HOME=/home/appuser/.cache/huggingface \
    PORT=8501

# Copy dependencies first so source-code changes can reuse this Docker layer.
COPY --chown=appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=appuser . .

# Build the index and download both Hugging Face models during image creation.
RUN python -m app.ingest
RUN python -c "from app.retrieval import _get_reranker; _get_reranker()"

EXPOSE 8501
ENTRYPOINT ["./entrypoint.sh"]
