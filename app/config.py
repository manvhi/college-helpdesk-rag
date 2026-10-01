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
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# RAG knobs -- these are the things you should experiment with and note the results
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
