"""Explicit Gemini context (prefix) caching.

Implicit prefix caching never hit for this model/endpoint (cached_content_token_count stayed empty),
so prompts whose large prefix is re-sent several times are cached explicitly:
  - research turn: the first-pass findings, shared by the reviewer (1-3 calls) and the synthesizer;
  - follow-up chat: the conversation up to the latest study brief, reused by every follow-up answer.
A cache stores system prompt + prefix; a call then sends only the tail. Any cache problem -> normal call.
"""
import logging
import threading
import time

from google.genai import types
from langchain_core.messages import AIMessage, SystemMessage

from .config import MODEL
from .embeddings import _client
from .guards import digest
from .llm import text_of

log = logging.getLogger("ml_researcher.cache")
MIN_CACHE_CHARS = 4096 * 4        # Vertex rejects explicit caches under 4096 tokens (~4 chars/token)
_caches: dict = {}                # key -> (cache name, expiry epoch)
_failed: set = set()              # keys whose create failed (e.g. under the model's minimum size): don't retry
_lock = threading.Lock()


def _content(msg) -> types.Content:
    return types.Content(role="model" if isinstance(msg, AIMessage) else "user",
                         parts=[types.Part(text=text_of(msg))])


def get_or_create(key: str, system: str, prefix: list, ttl: int) -> str | None:
    """Cache name for (system, prefix messages), creating it if needed; None if caching isn't possible."""
    with _lock:
        hit = _caches.get(key)
        if hit and hit[1] > time.time() + 30:               # keep a margin so it can't expire mid-call
            return hit[0]
        if key in _failed or len(system) + sum(len(text_of(m)) for m in prefix) < MIN_CACHE_CHARS:
            return None
        t0 = time.monotonic()
        try:
            c = _client.caches.create(model=MODEL, config=types.CreateCachedContentConfig(
                system_instruction=system, contents=[_content(m) for m in prefix], ttl=f"{ttl}s"))
        except Exception as e:
            _failed.add(key)
            log.warning("cache create failed: %s", str(e)[:200], extra={"json_fields": {
                "event": "cache_error", "key": key, "error": str(e)[:300]}})
            return None
        _caches[key] = (c.name, time.time() + ttl)
    tokens = getattr(c.usage_metadata, "total_token_count", None)
    log.info("cache created %s (%s tokens)", key, tokens, extra={"json_fields": {
        "event": "cache_create", "key": key, "tokens": tokens, "ttl_s": ttl,
        "duration_ms": round((time.monotonic() - t0) * 1000)}})
    return c.name


def drop(key: str) -> None:
    """Delete a cache early (best effort); it would expire on its own anyway."""
    with _lock:
        hit = _caches.pop(key, None)
    if hit:
        try:
            _client.caches.delete(name=hit[0])
        except Exception:
            pass


def invoke(llm, system: str, prefix: list, tail: list, key: str, ttl: int, schema=None):
    """llm.invoke([system] + prefix + tail), with system + prefix served from an explicit cache when possible.
    schema -> structured output. Falls back to the uncached call on any cache problem (e.g. expired)."""
    name = get_or_create(key, system, prefix, ttl) if prefix else None
    if name:
        model = llm.model_copy(update={"cached_content": name})
        try:
            return (model.with_structured_output(schema) if schema else model).invoke(tail)
        except Exception as e:
            with _lock:
                _caches.pop(key, None)
            log.warning("cached call failed, retrying uncached: %s", str(e)[:200], extra={"json_fields": {
                "event": "cache_error", "key": key, "error": str(e)[:300]}})
    return (llm.with_structured_output(schema) if schema else llm).invoke([SystemMessage(system)] + prefix + tail)


def key_for(kind: str, system: str, prefix: list) -> str:
    """Same system prompt + same prefix -> same key, so different calls find the same cache."""
    return f"{kind}:{digest([system] + [text_of(m) for m in prefix])}"
