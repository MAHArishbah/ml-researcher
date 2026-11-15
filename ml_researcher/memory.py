"""Long-term memory about the user, after Generative Agents (Park et al., 2023).

Two stores, both in the owner-only data folder:
  memories  a Chroma collection (ml_memory_<model>_<dim>): one short statement per memory, with importance (1-10),
            created / last_accessed times, kind (fact | preference | reflection | noted) and the conversation it
            came from. Written after every finished turn (after_turn, one cheap model call) and by /remember.
  profile   MEMORY_FILE (memory.json): a few stable key-value facts, editable by hand or with /remember key: value.

Recall (only when a message contains /memory) scores each candidate the way the paper does:
    score = w_rec * recency + w_imp * importance + w_rel * relevance,   each min-max scaled to [0, 1]
    recency = MEMORY_DECAY ** hours since last accessed;  importance = rating / 10;  relevance = cosine to the message
Candidates are the MEMORY_CANDIDATES nearest memories (all of them while there are fewer). Recalled memories get
last_accessed = now, so what keeps being useful stays fresh. Reflection: once the memories written since the last
reflection add up to MEMORY_REFLECT_AT importance, the model writes up to 3 insights that cite them; they are
stored as kind=reflection and recalled like any other memory.
"""
import json
import logging
import re
import threading
import time
import uuid
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .config import (EMBED_DIM, EMBED_MODEL, MEMORY_CANDIDATES, MEMORY_DECAY, MEMORY_FILE, MEMORY_REFLECT_AT,
                     MEMORY_TOP_K, MEMORY_WEIGHTS, private_dir)
from .embeddings import embed
from .llm import RESEARCH_LLM, structured
from .prompts import prompt
from .store_chroma import _chroma

log = logging.getLogger("ml_researcher.memory")
DUPLICATE_SIM = 0.92                         # a new memory this close to an old one refreshes it instead
EXCHANGE_CHARS = 6000                        # question + start of the reply sent to the extractor
_col, _lock = None, threading.RLock()


class Memory(BaseModel):
    text: str = Field(description="one short statement about the user, in the third person")
    importance: int = Field(ge=1, le=10)
    kind: Literal["fact", "preference"]


class ProfileFact(BaseModel):
    key: str = Field(description="snake_case key, e.g. learning_style")
    value: str


class Extracted(BaseModel):
    memories: list[Memory] = Field(default_factory=list, max_length=3)
    profile: list[ProfileFact] = Field(default_factory=list)


class Rating(BaseModel):
    importance: int = Field(ge=1, le=10)


class Insight(BaseModel):
    text: str
    importance: int = Field(ge=1, le=10)
    cites: list[int] = Field(description="numbers of the memories it rests on")


class Insights(BaseModel):
    insights: list[Insight] = Field(default_factory=list, max_length=3)


def _collection():
    global _col
    with _lock:
        if _col is None:
            _col = _chroma().get_or_create_collection(
                name=f"ml_memory_{EMBED_MODEL}_{EMBED_DIM}", configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None)
    return _col


# ---------- profile (key-value JSON) ----------

def profile() -> dict:
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}        # "_..." keys are bookkeeping


def _raw() -> dict:
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    private_dir(MEMORY_FILE.parent)
    MEMORY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_profile(key: str, value: str | None) -> None:
    """value None deletes the key."""
    with _lock:
        data = _raw()
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
        _save(data)


# ---------- memories ----------

def add(text: str, importance: int, kind: str, thread_id: str = "", sources: list | None = None) -> str:
    """Store one memory (or refresh a near-duplicate); returns its id. Checks whether it is time to reflect."""
    vector = embed(text, kind="doc").tolist()
    now = time.time()
    with _lock:
        col = _collection()
        if col.count():
            near = col.query(query_embeddings=[vector], n_results=1, include=["metadatas", "distances"])
            if near["ids"][0] and 1 - near["distances"][0][0] >= DUPLICATE_SIM:
                mid, meta = near["ids"][0][0], near["metadatas"][0][0]
                col.update(ids=[mid], metadatas=[{**meta, "last_accessed": now,
                                                  "importance": max(meta["importance"], importance)}])
                return mid
        mid = uuid.uuid4().hex[:12]
        col.add(ids=[mid], embeddings=[vector], documents=[text],
                metadatas=[{"importance": int(importance), "kind": kind, "created": now, "last_accessed": now,
                            "thread_id": thread_id, "sources": ",".join(sources or [])}])
        if kind != "reflection":                    # what reflection is about: events, not earlier insights
            data = _raw()
            data["_since_reflection"] = data.get("_since_reflection", 0) + int(importance)
            _save(data)
    log.info("memory added", extra={"json_fields": {"event": "memory_write", "memory_id": mid, "kind": kind,
                                                    "importance": importance}})
    if kind != "reflection" and _raw().get("_since_reflection", 0) >= MEMORY_REFLECT_AT:
        reflect()
    return mid


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.ones_like(x)


def recall(query: str, k: int = MEMORY_TOP_K) -> list:
    """The k memories with the highest recency + importance + relevance (each min-max scaled over the
    candidates, as in the paper); refreshes their last_accessed."""
    with _lock:
        col = _collection()
        n = col.count()
        if not n or not query.strip():
            return []
        res = col.query(query_embeddings=[embed(query, kind="query").tolist()], n_results=min(n, MEMORY_CANDIDATES),
                        include=["documents", "metadatas", "distances"])
        ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
        now = time.time()
        rec = np.array([MEMORY_DECAY ** ((now - m["last_accessed"]) / 3600) for m in metas])
        imp = np.array([m["importance"] / 10 for m in metas])
        rel = np.array([1 - d for d in res["distances"][0]])
        w_rec, w_imp, w_rel = MEMORY_WEIGHTS
        score = w_rec * _minmax(rec) + w_imp * _minmax(imp) + w_rel * _minmax(rel)
        top = np.argsort(-score)[:k]
        col.update(ids=[ids[i] for i in top], metadatas=[{**metas[i], "last_accessed": now} for i in top])
    out = [{"id": ids[i], "text": docs[i], "kind": metas[i]["kind"], "importance": metas[i]["importance"],
            "created": metas[i]["created"], "score": round(float(score[i]), 3),
            "recency": round(float(rec[i]), 3), "relevance": round(float(rel[i]), 3)} for i in top]
    log.info("memory recalled", extra={"json_fields": {"event": "memory_recall", "candidates": len(ids),
                                                       "recalled": [(m["id"], m["score"]) for m in out]}})
    return out


def context(query: str) -> tuple[str, list]:
    """The /memory block for a turn: the whole profile + the recalled memories. ("", []) when there are none."""
    facts, memories = profile(), recall(query)
    if not facts and not memories:
        return "", []
    lines = [f"- {k}: {v}" for k, v in facts.items()]
    days = lambda t: f"{(time.time() - t) / 86400:.0f} d ago"                       # noqa: E731
    lines += [f"- [{m['kind']}, importance {m['importance']}, {days(m['created'])}] {m['text']}" for m in memories]
    block = ("WHAT I REMEMBER ABOUT YOU (from earlier conversations):\n<memory>\n" + "\n".join(lines)
             + "\n</memory>\n" + prompt("MEMORY_USE"))
    return block, [{"id": m["id"], "text": m["text"], "score": m["score"]} for m in memories]


def after_turn(question: str, reply: str, thread_id: str) -> None:
    """Mine one finished exchange for lasting facts about the user (one model call); never raises."""
    try:
        exchange = f"USER:\n{question}\n\nASSISTANT (start of the reply):\n{reply}"[:EXCHANGE_CHARS]
        out = structured(RESEARCH_LLM, Extracted).invoke([("system", prompt("MEMORY_EXTRACT")), ("human", exchange)])
        for m in out.memories:
            add(m.text, m.importance, m.kind, thread_id)
        for f in out.profile:
            set_profile(f.key, f.value)
    except Exception as e:                           # memory is a bonus: a failure must not touch the turn
        log.warning("memory extraction failed: %s", e, extra={"json_fields": {
            "event": "memory_error", "thread_id": thread_id, "error": f"{type(e).__name__}: {str(e)[:200]}"}})


def reflect() -> None:
    """Insights from the memories written since the last reflection; resets the importance counter."""
    with _lock:
        data = _raw()
        since = data.get("_last_reflection", 0)
        got = _collection().get(where={"created": {"$gt": since}}, include=["documents", "metadatas"])
        data["_since_reflection"], data["_last_reflection"] = 0, time.time()
        _save(data)
    recent = [(i, d) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
              if m["kind"] != "reflection"][-50:]
    if len(recent) < 3:
        return
    numbered = "\n".join(f"{n}. {d}" for n, (_, d) in enumerate(recent, 1))
    try:
        out = structured(RESEARCH_LLM, Insights).invoke([("system", prompt("MEMORY_REFLECT")), ("human", numbered)])
    except Exception as e:
        log.warning("reflection failed: %s", e, extra={"json_fields": {"event": "memory_error", "error": str(e)[:200]}})
        return
    for ins in out.insights:
        cites = [recent[c - 1][0] for c in ins.cites if 1 <= c <= len(recent)]
        add(ins.text, ins.importance, "reflection", sources=cites)
    log.info("reflected", extra={"json_fields": {"event": "memory_reflect", "from": len(recent),
                                                 "insights": len(out.insights)}})


def command(text: str) -> str | None:
    """/remember <fact> or /remember key: value. Returns the confirmation to show, or None if `text` is not a
    memory command (then it is a normal message)."""
    m = re.match(r"\s*/remember\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    body = m.group(1).strip()
    kv = re.fullmatch(r"([A-Za-z][\w ]{0,39}):\s*(.+)", body, re.DOTALL)
    if kv and "\n" not in body:
        key = kv.group(1).strip().lower().replace(" ", "_")
        set_profile(key, kv.group(2).strip())
        return f"Profile updated: **{key}** = {kv.group(2).strip()}"
    try:
        importance = structured(RESEARCH_LLM, Rating).invoke(
            [("system", prompt("MEMORY_IMPORTANCE")), ("human", body)]).importance
    except Exception:
        importance = 5
    add(body, importance, "noted")
    return f"Remembered (importance {importance}/10): {body}"


def all_memories() -> list:
    """Every memory, newest first, for the sidebar."""
    got = _collection().get(include=["documents", "metadatas"])
    rows = [{"id": i, "text": d, **m} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)]
    return sorted(rows, key=lambda r: -r["created"])


def delete(memory_id: str) -> None:
    _collection().delete(ids=[memory_id])


def forget_thread(thread_id: str) -> None:
    """Delete the memories that came from one conversation (when that conversation is deleted)."""
    _collection().delete(where={"thread_id": thread_id})
