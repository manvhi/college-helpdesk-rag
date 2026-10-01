"""Step 1 of RAG: load documents -> split into chunks -> embed -> store in Chroma.

Run:  python -m app.ingest
Try:  python -m app.ingest --chunk-size 400 --chunk-overlap 80   (then compare answer quality)
"""
import argparse
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.rag import get_embeddings


def load_documents(data_dir=config.DATA_DIR):
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.name.lower().startswith("readme"):
            continue                                   # skip instructions files
        if path.suffix.lower() == ".pdf":
            pages = PyPDFLoader(str(path)).load()      # one Document per page
        elif path.suffix.lower() in {".txt", ".md"}:
            pages = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for p in pages:
            p.metadata["source"] = path.name           # keep just the file name for citations
        docs.extend(pages)
        print(f"  loaded {path.name}: {len(pages)} page(s)")
    return docs


def split_documents(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_index(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP, embeddings=None):
    print(f"Loading documents from {config.DATA_DIR} ...")
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found. Add PDF/TXT/MD files to the data/ folder first.")
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    print(f"{len(docs)} page(s) -> {len(chunks)} chunk(s) (size={chunk_size}, overlap={chunk_overlap})")

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)               # rebuild from scratch each time
    Chroma.from_documents(
        chunks,
        embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Done. Vector store saved to {config.CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()
    build_index(args.chunk_size, args.chunk_overlap)
