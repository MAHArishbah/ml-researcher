"""Cohere reranker: re-orders the fused vector + BM25 candidates by query/passage relevance."""
import logging
import threading
import time

from .config import COHERE_API_KEY, RERANK_MODEL

log = logging.getLogger("ml_researcher.rerank")
_client, _lock = None, threading.Lock()
RETRIES = 4


def available() -> bool:
    return bool(COHERE_API_KEY)


def _cohere():
    global _client
    with _lock:
        if _client is None:
            import cohere
            _client = cohere.ClientV2(api_key=COHERE_API_KEY, timeout=30)
    return _client


def _is_retryable(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    return code == 429 or (code is not None and code >= 500) or code is None   # None = network/timeout


def rerank(query: str, docs: list, top_n: int) -> list | None:
    """[(index into docs, relevance score)] best first, or None if reranking failed (caller keeps its order)."""
    last_err = None
    for attempt in range(RETRIES):
        try:
            res = _cohere().rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=top_n)
            return [(r.index, round(r.relevance_score, 4)) for r in res.results]
        except Exception as e:
            last_err = e
            if not _is_retryable(e) or attempt == RETRIES - 1:
                break
            time.sleep(2 ** attempt * (5 if getattr(e, "status_code", None) == 429 else 1))
    err = f"{getattr(last_err, 'status_code', None) or type(last_err).__name__}: {getattr(last_err, 'body', last_err)}"
    log.warning("rerank failed, using fused order: %s", err[:300],
                extra={"json_fields": {"event": "rerank_error", "error": err[:300]}})
    return None
