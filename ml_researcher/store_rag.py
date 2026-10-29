"""Google-managed backend: Vertex AI RAG Engine (RagManagedDb)."""
import tempfile
import threading
from pathlib import Path

import vertexai
from vertexai import rag

from .config import PROJECT, RAG_CORPUS_NAME, RAG_LOCATION
from .readers import read_file

_corpus, _lock = None, threading.Lock()


def _corpus_name() -> str:
    global _corpus
    with _lock:
        if _corpus is None:
            vertexai.init(project=PROJECT, location=RAG_LOCATION)
            _corpus = next((c.name for c in rag.list_corpora()
                            if c.display_name == RAG_CORPUS_NAME), None)
            if _corpus is None:
                _corpus = rag.create_corpus(display_name=RAG_CORPUS_NAME).name
    return _corpus


def _files():
    return list(rag.list_files(corpus_name=_corpus_name()))


def _key(f):
    """description stores 'path|mtime' -> (path, mtime)."""
    path, _, mtime = (f.description or "").rpartition("|")
    return path, mtime


def count() -> int:
    return len(_files())


def indexed_files() -> dict:
    return {path: float(mtime) for path, mtime in map(_key, _files()) if path}


def delete_paths(paths) -> None:
    paths = set(paths)
    for f in _files():
        if _key(f)[0] in paths:
            rag.delete_file(name=f.name)


def upsert_file(p: Path, mtime: float) -> int:
    delete_paths([str(p)])
    src, tmp = p, None
    if p.suffix.lower() == ".doc":                    # legacy .doc -> upload extracted text
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(read_file(p))
        src = Path(tmp.name)
    try:
        rag.upload_file(corpus_name=_corpus_name(), path=str(src),
                        display_name=p.name, description=f"{p}|{mtime}")
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)
    return 1


def search(query: str, top_k: int = 6, file_type: str = "") -> list:
    ext = file_type.lower().lstrip(".")
    resp = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=_corpus_name())],
        text=query,
        rag_retrieval_config=rag.RagRetrievalConfig(top_k=top_k * 3 if ext else top_k),
    )
    hits = []
    for c in resp.contexts.contexts:
        name = getattr(c, "source_display_name", "") or c.source_uri
        if ext and not name.lower().endswith("." + ext):
            continue
        hits.append({"name": name, "path": c.source_uri,
                     "score": getattr(c, "score", None), "text": c.text[:700]})
    return hits[:top_k]