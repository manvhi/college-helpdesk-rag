"""Step 1 of RAG: load documents -> split into PARENT and CHILD chunks -> embed children -> store.

Parent-document strategy:
  * PARENT chunks (large, ~2000 chars) keep enough surrounding text for the LLM to answer well.
  * CHILD chunks (small, ~400 chars) are cut from each parent. ONLY children are embedded and searched,
    because small chunks match a question precisely.
  * At question time we search children, then swap each hit for its parent (see app/retrieval.py).

Where things are stored:
  * children + embeddings -> Chroma (chroma_db/)
  * parents (plain text)  -> chroma_db/parents.json   (a simple dict: parent_id -> text, source, page)

Run:  python -m app.ingest
Try:  python -m app.ingest --child-size 300 --parent-size 1500
"""
import argparse
import json
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.rag import get_embeddings


def load_documents(data_dir=config.DATA_DIR):
    """Reads every PDF / .txt / .md recursively under data/."""
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


def make_parents_and_children(docs, parent_size, parent_overlap, child_size, child_overlap):
    """Returns (parents, children).
    parents  = {"p0": {"text":..., "source":..., "page":...}, ...}
    children = list of small Documents, each carrying metadata["parent_id"] that points into `parents`.
    Parents are cut from single pages, so a parent never mixes two pages (citations stay correct).
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=parent_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap)

    parents, children = {}, []
    for i, parent in enumerate(parent_splitter.split_documents(docs)):
        pid = f"p{i}"
        parents[pid] = {
            "text": parent.page_content,
            "source": parent.metadata.get("source"),
            "page": parent.metadata.get("page", 0),
        }
        for child in child_splitter.split_documents([parent]):   # children inherit source + page
            child.metadata["parent_id"] = pid
            children.append(child)
    return parents, children


def build_index(parent_size=config.PARENT_CHUNK_SIZE, parent_overlap=config.PARENT_CHUNK_OVERLAP,
                child_size=config.CHILD_CHUNK_SIZE, child_overlap=config.CHILD_CHUNK_OVERLAP,
                embeddings=None):
    print(f"Loading documents from {config.DATA_DIR} ...")
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found. Add PDF/TXT/MD files to the data/ folder first.")

    parents, children = make_parents_and_children(docs, parent_size, parent_overlap, child_size, child_overlap)
    print(f"{len(docs)} page(s) -> {len(parents)} parent chunk(s) (size={parent_size}) "
          f"-> {len(children)} child chunk(s) (size={child_size}, overlap={child_overlap})")

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)               # rebuild from scratch each time
    Chroma.from_documents(
        children,                                      # only the small children are embedded
        embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    config.PARENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.PARENTS_FILE.write_text(json.dumps(parents, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Children saved to {config.CHROMA_DIR}, parents saved to {config.PARENTS_FILE}")
    return len(children)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-size", type=int, default=config.PARENT_CHUNK_SIZE)
    ap.add_argument("--parent-overlap", type=int, default=config.PARENT_CHUNK_OVERLAP)
    ap.add_argument("--child-size", type=int, default=config.CHILD_CHUNK_SIZE)
    ap.add_argument("--child-overlap", type=int, default=config.CHILD_CHUNK_OVERLAP)
    a = ap.parse_args()
    build_index(a.parent_size, a.parent_overlap, a.child_size, a.child_overlap)
