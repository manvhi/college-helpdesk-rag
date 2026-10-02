FROM python:3.11-slim

# Hugging Face Spaces runs containers as user id 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch keeps the image much smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Build the vector index at image build time (also downloads the embedding model once)
RUN python -m app.ingest

EXPOSE 7860
# GOOGLE_API_KEY is provided at runtime as a Space secret
CMD ["sh", "start.sh"]
