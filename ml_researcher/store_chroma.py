"""Local backend: ChromaDB (HNSW, cosine) + optional BM25 hybrid search + Cohere rerank.

One Chroma database (CHROMA_DIR), one collection per index:
  NOTES  ml_notes_<model>_<dim>  your notes and books   (module functions below: count, search, ...)
  code   ml_code_<model>_<dim>   your own code          (codeindex.py builds a second ChromaIndex)
Each index has its own BM25 keyword index, so code and notes never compete for the same top-k slots.
"""
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb

from . import rerank as reranker
from .config import (CHROMA_DIR, EMBED_DIM, EMBED_MODEL, EMBED_WORKERS, HNSW_EF_SEARCH, HYBRID, RERANK,
                     RERANK_CANDIDATES)
from .embeddings import embed
from .hybrid import KeywordIndex, rrf
from .readers import chunks, read_file

GET_PAGE = 5000                   # one unpaged get() fails past ~32k chunks (SQLite "too many SQL variables")
EXTRA_FIELDS = ("symbol", "line", "lang", "rel_path")   # optional per-chunk metadata (code index) returned by search
_client, _client_lock = None, threading.Lock()


def _chroma():
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


class ChromaIndex:
    """One collection + its BM25 index.
    pieces(path) -> [(text, extra metadata)] decides how a file is split; title(path, text, extra) is the
    document prefix that is embedded (the model places a query near the passage that answers it)."""

    def __init__(self, kind: str, pieces, title):
        self.name = f"ml_{kind}_{EMBED_MODEL}_{EMBED_DIM}"          # new model/dim -> new collection
        self.pieces, self.title = pieces, title
        self._col, self._kw = None, None                            # collection; (count, KeywordIndex, store)
        self._col_lock, self._kw_lock = threading.Lock(), threading.Lock()

    def collection(self):
        with self._col_lock:
            if self._col is None:
                self._col = _chroma().get_or_create_collection(
                    name=self.name,
                    configuration={"hnsw": {"space": "cosine", "ef_construction": 200,
                                            "max_neighbors": 16, "ef_search": HNSW_EF_SEARCH}},
                    embedding_function=None)                        # we supply Gemini vectors
                try:
                    self._col.modify(configuration={"hnsw": {"ef_search": HNSW_EF_SEARCH}})
                except Exception:
                    pass
        return self._col

    def count(self) -> int:
        return self.collection().count()

    def get_all(self, include: list) -> dict:
        """col.get() of every chunk, fetched in pages. Returns {"ids": [...], <field>: [...] for each include}."""
        col, out, offset = self.collection(), {"ids": [], **{f: [] for f in include}}, 0
        while True:
            page = col.get(include=include, limit=GET_PAGE, offset=offset)
            for f in out:
                out[f].extend(page[f])
            if len(page["ids"]) < GET_PAGE:
                return out
            offset += GET_PAGE

    def indexed_files(self) -> dict:
        """path -> mtime for every stored file."""
        return {m["path"]: m["mtime"] for m in self.get_all(["metadatas"])["metadatas"]}

    def delete_paths(self, paths) -> None:
        paths = list(paths)
        if paths:
            self.collection().delete(where={"path": {"$in": paths}})
            self._kw = None

    def upsert_file(self, p: Path, mtime: float) -> int:
        """(Re)index one file. Returns chunks stored; 0 means no extractable text."""
        pieces = list(self.pieces(p))
        if pieces:
            with ThreadPoolExecutor(EMBED_WORKERS) as ex:
                vecs = list(ex.map(lambda tx: embed(self.title(p, *tx), kind="doc").tolist(), pieces))
        col = self.collection()
        col.delete(where={"path": str(p)})
        self._kw = None
        if not pieces:
            return 0
        fid = hashlib.sha1(str(p).encode()).hexdigest()[:16]
        col.upsert(
            ids=[f"{fid}:{j}" for j in range(len(pieces))],
            embeddings=vecs,
            documents=[t for t, _ in pieces],
            metadatas=[{"path": str(p), "name": p.name, "ext": p.suffix.lower().lstrip("."), "chunk": j,
                        "mtime": mtime, **extra} for j, (_, extra) in enumerate(pieces)])
        return len(pieces)

    def keyword_index(self):
        """BM25 over all stored chunks; rebuilt only when the chunk count changes."""
        col = self.collection()
        with self._kw_lock:
            n = col.count()
            if self._kw is None or self._kw[0] != n:
                data = self.get_all(["documents", "metadatas"])
                store = dict(zip(data["ids"], zip(data["documents"], data["metadatas"], strict=True), strict=True))
                self._kw = (n, KeywordIndex(data["ids"], data["documents"]), store)
            return self._kw[1], self._kw[2]

    def search(self, query: str, top_k: int = 5, file_type: str = "") -> list:
        """Vector + BM25 (RRF-fused) -> top RERANK_CANDIDATES -> Cohere rerank -> top_k full chunks.
        Without a Cohere key (or if the rerank call fails) the fused order is kept.
        HYBRID / RERANK are read from this module on every call (the eval switches them)."""
        col = self.collection()
        ext = file_type.lower().lstrip(".")
        use_rerank = RERANK and reranker.available()
        n_cands = max(top_k, RERANK_CANDIDATES) if use_rerank else top_k
        pool = min(max(n_cands * 2, 20), col.count())              # candidates per retriever before fusion
        if pool == 0:
            return []
        res = col.query(query_embeddings=[embed(query, kind="query").tolist()],
                        n_results=pool, where={"ext": ext} if ext else None,
                        include=["documents", "metadatas", "distances"])
        vec_ids = res["ids"][0]
        lookup = {i: (d, m) for i, d, m in zip(vec_ids, res["documents"][0], res["metadatas"][0], strict=True)}
        sims = {i: round(1 - dist, 3) for i, dist in zip(vec_ids, res["distances"][0], strict=True)}
        kw_ids = []
        if HYBRID:
            kw, store = self.keyword_index()
            kw_ids = [i for i in kw.top(query, pool) if not ext or store[i][1]["ext"] == ext]
            for i in kw_ids:
                lookup.setdefault(i, store[i])
            ranked = rrf([vec_ids, kw_ids])[:n_cands]
        else:
            ranked = vec_ids[:n_cands]

        scores = {}
        if use_rerank and len(ranked) > 1:
            order = reranker.rerank(query, [f"{lookup[i][1]['name']}\n{lookup[i][0]}" for i in ranked], top_k)
            if order:
                scores = {ranked[j]: s for j, s in order}
                ranked = [ranked[j] for j, _ in order]
        ranked = ranked[:top_k]

        kw_set = set(kw_ids)
        return [{"id": i, "name": lookup[i][1]["name"], "path": lookup[i][1]["path"], "chunk": lookup[i][1]["chunk"],
                 **{f: lookup[i][1][f] for f in EXTRA_FIELDS if f in lookup[i][1]},
                 "vector_similarity": sims.get(i), "keyword_hit": i in kw_set, "rerank_score": scores.get(i),
                 "text": lookup[i][0]} for i in ranked]


# ---------- the notes index, with the module-level API the rest of the code (and the eval) uses ----------
NOTES = ChromaIndex("notes", pieces=lambda p: [(t, {}) for t in chunks(read_file(p))],
                    title=lambda p, t, extra: f"title: {p.name} | text: {t}")


def _collection():
    return NOTES.collection()


def count() -> int:
    return NOTES.count()


def get_all(include: list) -> dict:
    return NOTES.get_all(include)


def indexed_files() -> dict:
    return NOTES.indexed_files()


def delete_paths(paths) -> None:
    NOTES.delete_paths(paths)


def upsert_file(p: Path, mtime: float) -> int:
    return NOTES.upsert_file(p, mtime)


def _keyword_index():
    return NOTES.keyword_index()


def search(query: str, top_k: int = 5, file_type: str = "") -> list:
    return NOTES.search(query, top_k, file_type)
