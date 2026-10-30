"""Prompt (prefix) caching behind one interface, with one adapter per provider.

Where a large prefix is re-sent several times:
  - research turn: the first-pass findings, shared by the reviewer (1-3 calls) and the synthesizer;
  - follow-up chat: the conversation up to the latest study brief, reused by every follow-up answer.

  google_genai  explicit context caches (implicit caching never hit for this model/endpoint); the cache holds
                system + prefix and calls send only the tail. Vertex needs >= 4096 tokens.
  anthropic     cache_control breakpoints on the system prompt and the last prefix message; Claude caches
                the prefix server-side (>= ~1-4k tokens depending on model, smaller prefixes just aren't cached).
  anything else no caching, plain call.
Any cache problem falls back to the plain call.
"""
import logging
import threading
import time

from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import ratelimit
from .embeddings import _client
from .guards import digest
from .llm import provider_of, structured, text_of

log = logging.getLogger("ml_researcher.cache")
MIN_CACHE_CHARS = 4096 * 4        # Vertex rejects explicit caches under 4096 tokens (~4 chars/token)
_caches: dict = {}                # key -> (cache name, expiry epoch)       (Gemini explicit caches)
_failed: set = set()              # keys whose create failed (e.g. under the model's minimum size): don't retry
_lock = threading.Lock()


def _plain(llm, schema):
    return structured(llm, schema) if schema else llm


# ---------- Gemini: explicit context caches ----------
def _content(msg) -> types.Content:
    return types.Content(role="model" if isinstance(msg, AIMessage) else "user",
                         parts=[types.Part(text=text_of(msg))])


def get_or_create(model: str, key: str, system: str, prefix: list, ttl: int) -> str | None:
    """Gemini cache name for (system, prefix messages), creating it if needed; None if caching isn't possible."""
    with _lock:
        hit = _caches.get(key)
        if hit and hit[1] > time.time() + 30:               # keep a margin so it can't expire mid-call
            return hit[0]
        if key in _failed or len(system) + sum(len(text_of(m)) for m in prefix) < MIN_CACHE_CHARS:
            return None
        t0 = time.monotonic()
        try:
            ratelimit.wait()
            c = _client.caches.create(model=model, config=types.CreateCachedContentConfig(
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


def _gemini_invoke(llm, system, prefix, tail, key, ttl, schema):
    name = get_or_create(llm.model, key, system, prefix, ttl) if prefix else None
    if name:
        try:
            return _plain(llm.model_copy(update={"cached_content": name}), schema).invoke(tail)
        except Exception as e:
            with _lock:
                _caches.pop(key, None)
            log.warning("cached call failed, retrying uncached: %s", str(e)[:200], extra={"json_fields": {
                "event": "cache_error", "key": key, "error": str(e)[:300]}})
    return _plain(llm, schema).invoke([SystemMessage(system)] + prefix + tail)


# ---------- Claude: cache_control breakpoints ----------
_EPHEMERAL = {"type": "ephemeral"}


def _marked(msg):
    """Same message with a cache breakpoint at its end (everything up to here gets cached)."""
    cls = AIMessage if isinstance(msg, AIMessage) else HumanMessage
    return cls(content=[{"type": "text", "text": text_of(msg), "cache_control": _EPHEMERAL}])


def _anthropic_invoke(llm, system, prefix, tail, schema):
    sys_msg = SystemMessage(content=[{"type": "text", "text": system, "cache_control": _EPHEMERAL}])
    prefix = prefix[:-1] + [_marked(prefix[-1])] if prefix else []
    return _plain(llm, schema).invoke([sys_msg] + prefix + tail)


# ---------- interface ----------
def invoke(llm, system: str, prefix: list, tail: list, key: str, ttl: int, schema=None):
    """llm.invoke([system] + prefix + tail) with system + prefix cached the provider's way when possible.
    schema -> structured output."""
    provider = provider_of(llm)
    if provider == "google_genai":
        return _gemini_invoke(llm, system, prefix, tail, key, ttl, schema)
    if provider == "anthropic":
        return _anthropic_invoke(llm, system, prefix, tail, schema)
    return _plain(llm, schema).invoke([SystemMessage(system)] + prefix + tail)


def drop(key: str) -> None:
    """Delete a Gemini cache early (best effort); it would expire on its own anyway. No-op for other providers."""
    with _lock:
        hit = _caches.pop(key, None)
    if hit:
        try:
            _client.caches.delete(name=hit[0])
        except Exception:
            pass


def key_for(kind: str, system: str, prefix: list) -> str:
    """Same system prompt + same prefix -> same key, so different calls find the same cache."""
    return f"{kind}:{digest([system] + [text_of(m) for m in prefix])}"
