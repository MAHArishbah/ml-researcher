"""Local backend: ChromaDB (HNSW, cosine) + optional BM25 hybrid search."""
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

_col, _kw = None, None            # collection; (chunk_count, KeywordIndex, {id: (doc, meta)})
_col_lock, _kw_lock = threading.Lock(), threading.Lock()

def _collection():
    global _col
    with _col_lock:
        if _col is None:
            client=chromadb.PersistentClient(path=str(CHROMA_DIR))
            _col = client.get_or_create_collection(
                    name=f"ml_notes_{EMBED_MODEL}_{EMBED_DIM}",     # new model/dim -> new collection
                    configuration={"hnsw": {"space": "cosine", "ef_construction": 200,
                                            "max_neighbors": 16, "ef_search": HNSW_EF_SEARCH}},
                    embedding_function=None,                       # we supply Gemini vectors
                )
            try:
                _col.modify(configuration={"hnsw":{"ef_search":HNSW_EF_SEARCH}})
            except Exception:
                pass
    return _col

def count()->int:
    return _collection().count()

GET_PAGE = 5000                   # one unpaged get() fails past ~32k chunks (SQLite "too many SQL variables")

def get_all(include:list)->dict:
    """col.get() of every chunk, fetched in pages. Returns {"ids": [...], <field>: [...] for each include}."""
    col=_collection()
    out={"ids":[],**{f:[] for f in include}}
    offset=0
    while True:
        page=col.get(include=include,limit=GET_PAGE,offset=offset)
        for f in out:
            out[f].extend(page[f])
        if len(page["ids"])<GET_PAGE:
            return out
        offset+=GET_PAGE

def indexed_files()->dict:
    """path -> mtime for every stored file."""
    return {m["path"]:m["mtime"] for m in get_all(["metadatas"])["metadatas"]}

def delete_paths(paths) -> None:
    global _kw
    paths = list(paths)
    if paths:
        _collection().delete(where={"path": {"$in": paths}})        
        _kw=None

def upsert_file(p:Path,mtime:float)->int:
    """(Re)index one file. Returns chunks stored; 0 means no extractable text."""
    global _kw
    pieces=list(chunks(read_file(p)))
    if pieces:
        with ThreadPoolExecutor(EMBED_WORKERS) as ex:
            vecs=list(ex.map(
                lambda t:embed(f"title: {p.name} | text: {t}",kind='doc').tolist(),pieces
            ))
    col=_collection()
    col.delete(where={"path":str(p)})
    _kw=None
    if not pieces:
        return 0
    fid=hashlib.sha1(str(p).encode()).hexdigest()[:16]
    col.upsert(
        ids=[f"{fid}:{j}" for j in range(len(pieces))],
        embeddings=vecs,
        documents=pieces,
        metadatas=[{"path": str(p), "name": p.name, "ext": p.suffix.lower().lstrip("."),
                    "chunk": j, "mtime": mtime} for j in range(len(pieces))],
    )
        
    return len(pieces)

def _keyword_index():
    """BM25 over all stored chunks; rebuilt only when the chunk count changes."""
    global _kw
    col = _collection()
    with _kw_lock:
        n = col.count()
        if _kw is None or _kw[0] != n:
            data = get_all(["documents", "metadatas"])
            store = dict(zip(data["ids"], zip(data["documents"], data["metadatas"], strict=True), strict=True))
            _kw = (n, KeywordIndex(data["ids"], data["documents"]), store)
        return _kw[1], _kw[2]

def search(query:str,top_k:int=5,file_type:str="")->list:
    """Vector + BM25 (RRF-fused) -> top RERANK_CANDIDATES -> Cohere rerank -> top_k full chunks.
    Without a Cohere key (or if the rerank call fails) the fused order is kept."""
    col=_collection()
    ext=file_type.lower().lstrip(".")
    use_rerank=RERANK and reranker.available()
    n_cands=max(top_k,RERANK_CANDIDATES) if use_rerank else top_k
    pool=min(max(n_cands*2,20),col.count()) #candidates per retriever before fusion
    if pool==0:
        return []
    res = col.query(query_embeddings=[embed(query, kind="query").tolist()],
                    n_results=pool, where={"ext": ext} if ext else None,
                    include=["documents", "metadatas", "distances"])
    vec_ids=res["ids"][0]
    lookup = {i: (d, m) for i, d, m in zip(vec_ids, res["documents"][0], res["metadatas"][0], strict=True)}
    sims = {i: round(1 - dist, 3) for i, dist in zip(vec_ids, res["distances"][0], strict=True)}
    kw_ids = []
    if HYBRID:
        kw, store = _keyword_index()
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
             "vector_similarity": sims.get(i), "keyword_hit": i in kw_set, "rerank_score": scores.get(i),
             "text": lookup[i][0]} for i in ranked]

