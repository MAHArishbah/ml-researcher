"""Saved conversations, beyond what LangGraph's checkpointer offers: keeping only what the app needs, message
search across every conversation, and finding / removing questions that never got an answer.

Storage: LangGraph's SqliteSaver keeps a full copy of the state at every step of every turn, forever (in Nov 2026:
55 MB of history for 1.3 MB of current state). prune() keeps each idle conversation's latest checkpoint only.
That is safe here: the SQLite saver stores channel values inline in each checkpoint and the graph uses no
DeltaChannel, so the latest row stands on its own. Lost: stepping back to older steps, which the app never does.

Search: a contentless SQLite FTS5 table (message_search) holds only the word index, never the text; search_rows
maps each index row to its message. Snippets are cut from the conversation itself, for the hits only. A
conversation is re-indexed when its latest checkpoint changes (search_meta). No model calls.

    python -m ml_researcher.conversations --compact     # once, with the app stopped: prune all + VACUUM
"""
import argparse
import re
import sqlite3
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, RemoveMessage

from .llm import text_of

SNIPPET_BEFORE, SNIPPET_AFTER = 6, 10        # words of context around the first match
STEM = 5                                     # snippet matching: words agreeing on their first 5 letters
IDLE_SECS = 600                              # start-up prune skips conversations saved more recently


# ---------- storage ----------

def prune(conn: sqlite3.Connection, thread_ids) -> int:
    """Keep only the latest top-level checkpoint of each conversation: drop older ones, every researcher sub-run
    checkpoint and their pending writes. Never call it on a conversation with a turn running. Returns rows
    removed; freed pages go back to the disk when the file uses incremental auto-vacuum (see --compact)."""
    removed = 0
    for t in thread_ids:
        latest = conn.execute("SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ''",
                              (t,)).fetchone()[0]
        if latest is None:
            continue
        keep = "thread_id = ? AND NOT (checkpoint_ns = '' AND checkpoint_id = ?)"
        removed += conn.execute(f"DELETE FROM checkpoints WHERE {keep}", (t, latest)).rowcount
        conn.execute(f"DELETE FROM writes WHERE {keep}", (t, latest))
    conn.commit()
    if removed:
        conn.execute("PRAGMA incremental_vacuum")
    return removed


def idle_threads(conn: sqlite3.Connection, graph, busy: set) -> list:
    """Conversations safe to prune: not running here, and not saved in the last IDLE_SECS (a turn may be
    running in the terminal chat, which the UI's registry doesn't see)."""
    now = datetime.now(timezone.utc)
    out = []
    for (t,) in conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall():
        saved = graph.get_state({"configurable": {"thread_id": t}}).created_at
        if t not in busy and saved and (now - datetime.fromisoformat(saved)).total_seconds() > IDLE_SECS:
            out.append(t)
    return out


def forget(conn: sqlite3.Connection, saver, thread_id: str) -> None:
    """Delete a conversation everywhere it is kept: checkpoints, search index, star."""
    _setup(conn)
    saver.delete_thread(thread_id)
    _unindex(conn, thread_id)
    conn.execute("DELETE FROM search_meta WHERE thread_id = ?", (thread_id,))
    conn.execute("CREATE TABLE IF NOT EXISTS starred (thread_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM starred WHERE thread_id = ?", (thread_id,))
    conn.commit()


# ---------- search ----------

def _setup(conn: sqlite3.Connection) -> None:
    old = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'message_search'").fetchone()
    if old and "content=''" not in old[0]:          # the first version kept a copy of every message's text
        conn.execute("DROP TABLE message_search")
        conn.execute("DROP TABLE IF EXISTS search_meta")
        conn.execute("DROP TABLE IF EXISTS search_rows")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5("
                 "text, content='', contentless_delete=1, tokenize='porter unicode61')")
    conn.execute("CREATE TABLE IF NOT EXISTS search_rows "
                 "(rowid INTEGER PRIMARY KEY, thread_id TEXT, idx INTEGER, role TEXT)")
    conn.execute("CREATE INDEX IF NOT EXISTS search_rows_thread ON search_rows (thread_id)")
    conn.execute("CREATE TABLE IF NOT EXISTS search_meta (thread_id TEXT PRIMARY KEY, checkpoint_id TEXT)")


def _unindex(conn: sqlite3.Connection, thread_id: str) -> None:
    conn.execute("DELETE FROM message_search WHERE rowid IN (SELECT rowid FROM search_rows WHERE thread_id = ?)",
                 (thread_id,))
    conn.execute("DELETE FROM search_rows WHERE thread_id = ?", (thread_id,))


def sync(conn: sqlite3.Connection, graph) -> int:
    """Bring the index up to date with the checkpoints; returns how many conversations were re-indexed."""
    _setup(conn)
    latest = dict(conn.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints "
                               "WHERE checkpoint_ns = '' GROUP BY thread_id").fetchall())
    indexed = dict(conn.execute("SELECT thread_id, checkpoint_id FROM search_meta").fetchall())
    stale = [t for t, cid in latest.items() if indexed.get(t) != cid]
    gone = [t for t in indexed if t not in latest]                  # deleted conversations
    for t in stale + gone:
        _unindex(conn, t)
        conn.execute("DELETE FROM search_meta WHERE thread_id = ?", (t,))
    for t in stale:
        messages = graph.get_state({"configurable": {"thread_id": t}}).values.get("messages", [])
        for n, m in enumerate(messages):
            row = conn.execute("INSERT INTO search_rows (thread_id, idx, role) VALUES (?, ?, ?)",
                               (t, n, "user" if isinstance(m, HumanMessage) else "assistant")).lastrowid
            conn.execute("INSERT INTO message_search (rowid, text) VALUES (?, ?)", (row, text_of(m)))
        conn.execute("INSERT INTO search_meta VALUES (?, ?)", (t, latest[t]))
    conn.commit()
    return len(stale)


def _fts_query(text: str) -> str:
    """The user's words as an FTS5 query. "Quoted phrases" and prefix* work; anything else is quoted word by
    word, so punctuation (C++, what's, p50/p95) can't break the query syntax."""
    parts = re.findall(r'"[^"]+"|\S+', text)
    out = []
    for p in parts:
        if p.startswith('"') and p.endswith('"'):
            out.append(p)
        else:
            star = p.endswith("*")
            words = re.findall(r"\w+", p)
            out += [f'"{w}"' + ("*" if star and i == len(words) - 1 else "") for i, w in enumerate(words)]
    return " ".join(out)


def _snippet(text: str, query: str) -> str:
    """The words around the first match, matches in **bold**. FTS5 stems words (batch ~ batching); matching
    the first STEM letters is a close enough stand-in here."""
    stems = {w[:STEM] for w in re.findall(r"\w+", query.lower())}
    words = text.split()
    hit = [i for i, w in enumerate(words) if any(re.sub(r"\W", "", w.lower()).startswith(s) for s in stems)]
    if not hit:
        return " ".join(words[:SNIPPET_BEFORE + SNIPPET_AFTER]) + " …"
    lo, hi = max(hit[0] - SNIPPET_BEFORE, 0), min(hit[0] + SNIPPET_AFTER, len(words))
    shown = [f"**{w}**" if i in hit else w for i, w in enumerate(words[lo:hi], start=lo)]
    return ("… " if lo else "") + " ".join(shown) + (" …" if hi < len(words) else "")


def search(conn: sqlite3.Connection, graph, text: str, thread_id: str | None = None, limit: int = 30) -> list:
    """Messages matching every word, best first: dicts with thread_id, idx, role and a snippet."""
    query = _fts_query(text)
    if not query:
        return []
    sql = ("SELECT r.thread_id, r.idx, r.role FROM message_search JOIN search_rows r "
           "ON r.rowid = message_search.rowid WHERE message_search MATCH ?"
           + (" AND r.thread_id = ?" if thread_id else "") + " ORDER BY rank LIMIT ?")
    rows = conn.execute(sql, [query] + ([thread_id] if thread_id else []) + [limit]).fetchall()
    convos = {t: graph.get_state({"configurable": {"thread_id": t}}).values.get("messages", [])
              for t in {r[0] for r in rows}}                        # each conversation read once
    return [{"thread_id": t, "idx": i, "role": role, "snippet": _snippet(text_of(convos[t][i]), text)}
            for t, i, role in rows if i < len(convos[t])]


# ---------- unanswered questions ----------

def unanswered(messages: list) -> list:
    """Positions of questions with no reply: followed by another question, or the last message. A turn leaves
    one behind when it dies without rolling back (the app restarted mid-turn, or two turns ran at once)."""
    return [n for n, m in enumerate(messages) if isinstance(m, HumanMessage)
            and (n + 1 == len(messages) or isinstance(messages[n + 1], HumanMessage))]


def remove_messages(graph, config: dict, positions: list) -> None:
    """Delete messages by position. summary_upto is a position too, so it moves back by the number removed
    before it; the summary text itself is left as written."""
    values = graph.get_state(config).values
    messages = values["messages"]
    update = {"messages": [RemoveMessage(id=messages[n].id) for n in positions]}
    if values.get("summary_upto"):
        update["summary_upto"] = values["summary_upto"] - sum(n < values["summary_upto"] for n in positions)
    graph.update_state(config, update, as_node="answer")          # answer -> END: the thread stays idle


# ---------- one-time compaction ----------

def compact() -> None:
    """Prune every conversation, check nothing visible changed, switch the file to incremental auto-vacuum and
    VACUUM it, make the data folder owner-only, and delete earlier backups of the checkpoint file."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    from .config import CHECKPOINT_DB, DATA_DIR
    from .graph import build_graph

    conn = sqlite3.connect(str(CHECKPOINT_DB))
    graph = build_graph(SqliteSaver(conn))
    size = CHECKPOINT_DB.stat().st_size
    threads = [t for (t,) in conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()]
    state = lambda t: graph.get_state({"configurable": {"thread_id": t}}).values     # noqa: E731
    before = {t: state(t) for t in threads}
    removed = prune(conn, threads)
    for t in threads:
        after = state(t)
        for key in ("messages", "summary_upto", "history_summary", "brief"):
            assert after.get(key) == before[t].get(key), f"{t}: {key} changed; restore a backup"
    print(f"pruned {removed} checkpoint rows in {len(threads)} conversations; every conversation unchanged")
    try:
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        conn.execute("VACUUM")                      # rewrites the file; needs nobody else using it
    except sqlite3.OperationalError as e:
        raise SystemExit(f"VACUUM failed ({e}): stop the app (streamlit) and run this again")
    conn.close()
    print(f"{CHECKPOINT_DB.name}: {size / 1e6:.1f} MB -> {CHECKPOINT_DB.stat().st_size / 1e6:.1f} MB")
    for p in [DATA_DIR, *DATA_DIR.rglob("*")]:
        p.chmod(0o700 if p.is_dir() else 0o600)
    print(f"{DATA_DIR}: owner-only (folders 700, files 600)")
    for b in DATA_DIR.glob("checkpoints.backup-*.sqlite"):
        b.unlink()
        print(f"deleted backup {b.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", action="store_true", help="prune every conversation and shrink the file")
    if ap.parse_args().compact:
        compact()
    else:
        ap.print_help()
