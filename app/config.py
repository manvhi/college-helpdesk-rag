"""All settings in one place. Override any of them via environment variables / .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"          # put your PDFs / .txt / .md files here
CHROMA_DIR = BASE_DIR / "chroma_db"   # vector database is stored here

# Local embedding model (free, no API key, downloads once ~90MB)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM used to write the final answer. Check Google AI Studio for currently available model names.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# Parent-document chunking: small CHILD chunks are searched, big PARENT chunks go to the LLM
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "60"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
PARENTS_FILE = CHROMA_DIR / "parents.json"            # parent text, saved by ingest

# Retrieval knobs
TOP_K = int(os.getenv("TOP_K", "3"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "rerank")
# Reranking: hybrid search fetches this many candidates, the cross-encoder keeps the best k
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "10"))