"""Researcher page (registered in ui.py): the chat with the research agent."""
import json
import queue
import sqlite3
import threading
import time

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ml_researcher import vectordb
from ml_researcher.config import CHECKPOINT_DB
from ml_researcher.graph import build_graph
from ml_researcher.llm import text_of
from ml_researcher.logging_config import setup_logging
from ml_researcher.render import to_streamlit_md
from ml_researcher.runner import REPLY_NODES, TurnCancelled, run_turn

setup_logging(app="ui")

STEP_LABELS = {
    "router": "Decided: research or answer",
    "planner": "Planned searches",
    "arxiv_researcher": "Searched arXiv",
    "notes_researcher": "Searched your notes",
    "github_researcher": "Searched GitHub",
    "youtube_researcher": "Searched YouTube",
    "code_researcher": "Searched your code",
    "web_researcher": "Web search (only for /web questions)",
    "reviewer": "Reviewed findings for gaps",
    "followup": "Follow-up search or Python check",
    "synthesizer": "Wrote study brief",
    "answer": "Answered from conversation",
}
FINDINGS = [("arxiv_findings", "arXiv"), ("local_findings", "My notes"), ("code_findings", "My code"),
            ("github_findings", "GitHub"), ("youtube_findings", "YouTube")]
REDRAW_SECS = 0.1                            # re-render markdown at most 10x per second


class StoppableSaver(SqliteSaver):
    """Shares the base saver's connection + lock, but refuses writes once `stop` is set,
    so a stopped run can't add anything to the thread after it has been rolled back."""

    def __init__(self, base: SqliteSaver, stop: threading.Event):
        super().__init__(base.conn, serde=base.serde)
        self.lock, self.is_setup, self.stop = base.lock, True, stop

    def put(self, *args, **kwargs):
        with self.lock:
            if self.stop.is_set():
                raise TurnCancelled()
            return super().put(*args, **kwargs)

    def put_writes(self, *args, **kwargs):
        with self.lock:
            if self.stop.is_set():
                raise TurnCancelled()
            return super().put_writes(*args, **kwargs)


@st.cache_resource
def backend():
    """One SQLite saver + compiled graph shared by all browser sessions."""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False))
    saver.lock = threading.RLock()           # re-entrant: StoppableSaver holds it around super().put
    return saver, build_graph(checkpointer=saver)


def list_threads(saver) -> list:
    """Thread ids, most recently used first (checkpoint ids are time-ordered)."""
    try:
        with saver.lock:
            rows = saver.conn.execute("SELECT thread_id FROM checkpoints GROUP BY thread_id "
                                      "ORDER BY MAX(checkpoint_id) DESC").fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:        # table not created before the first message
        return []


def starred_threads(saver) -> set:
    """Starred conversations: a small table in the same SQLite file as the checkpoints."""
    with saver.lock:
        saver.conn.execute("CREATE TABLE IF NOT EXISTS starred (thread_id TEXT PRIMARY KEY)")
        return {r[0] for r in saver.conn.execute("SELECT thread_id FROM starred")}


def set_star(saver, thread_id: str, on: bool):
    with saver.lock:
        if on:
            saver.conn.execute("INSERT OR IGNORE INTO starred VALUES (?)", (thread_id,))
        else:
            saver.conn.execute("DELETE FROM starred WHERE thread_id = ?", (thread_id,))
        saver.conn.commit()


def pick_thread(key: str):
    """A conversation picker changed: make its choice the current conversation."""
    if st.session_state.get(key):
        st.session_state.thread = st.session_state[key]


def run_in_background(turn_graph, thread_id: str, prompt: str, stop: threading.Event,
                      events: queue.Queue):
    """Worker thread: runs the turn and reports progress as (kind, value) events. No st.* calls here."""
    try:
        reply = run_turn(turn_graph, thread_id, prompt,
                         on_step=lambda node, s: events.put(("step", (node, s))),
                         on_token=lambda text: events.put(("token", text)),
                         should_stop=stop.is_set,
                         on_notice=lambda message: events.put(("notice", message)))
        events.put(("done", reply))
    except Exception as e:
        if not stop.is_set():
            events.put(("error", e))


def copy_button(text: str, key: str):
    """'Copy' menu with the raw text (markdown + LaTeX source) in a code block, which has a copy icon."""
    with st.popover("Copy", icon=":material/content_copy:", type="tertiary", key=key):
        st.code(text, language="markdown", wrap_lines=True)


# ↑ / ↓ in the chat box step through this conversation's earlier prompts (like a terminal). ↑ only acts with the
# caret on the first line and ↓ on the last, so multi-line editing still works; ↓ past the newest restores the draft.
PROMPT_HISTORY_JS = """
<script>
(() => {
  const w = window.parent && window.parent.document ? window.parent : window;
  w.__promptHistory = %s;
  w.__historyIndex = w.__promptHistory.length;
  if (w.__historyHooked) return;
  w.__historyHooked = true;
  const setValue = (box, text) => {
    Object.getOwnPropertyDescriptor(w.HTMLTextAreaElement.prototype, "value").set.call(box, text);
    box.dispatchEvent(new w.Event("input", {bubbles: true}));
    box.setSelectionRange(text.length, text.length);
  };
  w.document.addEventListener("keydown", (e) => {
    const box = e.target;
    if (!box.matches || !box.matches('[data-testid="stChatInputTextArea"]')) return;
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    if (e.shiftKey || e.altKey || e.ctrlKey || e.metaKey) return;
    const hist = w.__promptHistory, v = box.value;
    if (e.key === "ArrowUp") {
      if (v.slice(0, box.selectionStart).includes("\\n") || w.__historyIndex === 0 || !hist.length) return;
      if (w.__historyIndex === hist.length) w.__historyDraft = v;
      w.__historyIndex -= 1;
    } else {
      if (v.slice(box.selectionEnd).includes("\\n") || w.__historyIndex >= hist.length) return;
      w.__historyIndex += 1;
    }
    e.preventDefault();
    setValue(box, w.__historyIndex < hist.length ? hist[w.__historyIndex] : (w.__historyDraft || ""));
  }, true);
})();
</script>
"""


def stop_and_edit():
    """Stop button callback (runs at the start of the next script run): put the prompt back for editing."""
    st.session_state.chat_box = st.session_state.get("pending_prompt", "")


saver, graph = backend()
threads = list_threads(saver)
if "thread" not in st.session_state:
    st.session_state.thread = threads[0] if threads else "default"

# ---------- sidebar: conversations + run details ----------
with st.sidebar:
    st.title("🔬 ML Researcher")
    with st.form("new_thread", clear_on_submit=True, border=False):
        name = st.text_input("New conversation", placeholder="e.g. ggn-fisher")
        if st.form_submit_button("Start") and name.strip():
            st.session_state.thread = name.strip()
    stars = starred_threads(saver)
    options = threads if st.session_state.thread in threads else [st.session_state.thread] + threads
    starred, general = [t for t in options if t in stars], [t for t in options if t not in stars]  # newest first
    # Two pickers, one current conversation: the picker that holds it shows it, the other shows nothing.
    current = st.session_state.thread
    st.session_state.pick_starred = current if current in starred else None
    st.session_state.pick_general = current if current in general else None
    if starred:
        st.selectbox("★ Starred", starred, key="pick_starred", placeholder="Choose a starred conversation",
                     on_change=pick_thread, args=("pick_starred",))
    st.selectbox("Conversations", general, key="pick_general", placeholder="Choose a conversation",
                 on_change=pick_thread, args=("pick_general",))
    is_starred = st.session_state.thread in stars
    if st.button("★ Starred" if is_starred else "☆ Star", key="star", type="tertiary",
                 help="Unstar this conversation" if is_starred else "Star this conversation: it moves to the top"):
        set_star(saver, st.session_state.thread, not is_starred)
        st.rerun()
    try:
        st.caption(f"{vectordb.count():,} chunks indexed")
    except Exception as e:
        st.caption(f"index unavailable: {e}")

config = {"configurable": {"thread_id": st.session_state.thread}}
values = graph.get_state(config).values

with st.sidebar:
    if values.get("topic"):
        st.divider()
        st.markdown(f"**Last topic:** {to_streamlit_md(values['topic'])}")
        if plan := values.get("search_plan"):
            with st.expander("Search queries"):
                for source, query in plan.items():
                    st.markdown(f"**{source}:** `{query}`")
        for key, label in FINDINGS:
            if values.get(key):
                with st.expander(f"Raw findings: {label}"):
                    st.markdown(to_streamlit_md(values[key]))

# ---------- chat ----------
st.subheader(f"Conversation: {st.session_state.thread}")
messages = values.get("messages", [])
for n, m in enumerate(messages):
    with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
        st.markdown(to_streamlit_md(text_of(m)))
        copy_button(text_of(m), key=f"copy_{st.session_state.thread}_{n}")

if messages and not isinstance(messages[-1], HumanMessage):
    st.download_button("Download last reply (.md)", text_of(messages[-1]),
                       file_name=f"{st.session_state.thread}.md", mime="text/markdown")

past_prompts = [text_of(m) for m in messages if isinstance(m, HumanMessage)]
st.html(PROMPT_HISTORY_JS % json.dumps(past_prompts).replace("</", "<\\/"), unsafe_allow_javascript=True)

if prompt := st.chat_input("Ask about an ML topic, or follow up on a brief", key="chat_box"):
    st.session_state.pending_prompt = prompt
    with st.chat_message("user"):
        st.markdown(to_streamlit_md(prompt))
    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)
        reply = st.empty()
        stop_slot = st.empty()
        stop_slot.button("⏹ Stop", key="stop", on_click=stop_and_edit,
                         help="Stop generating and put the prompt back in the box to edit")

        before = graph.get_state(config)
        pre_turn = before.config if before.created_at else None    # None -> brand-new thread
        stop, events = threading.Event(), queue.Queue()
        turn_graph = graph.copy({"checkpointer": StoppableSaver(saver, stop)})
        threading.Thread(target=run_in_background, daemon=True,
                         args=(turn_graph, st.session_state.thread, prompt, stop, events)).start()

        text, final, failed, last_draw = "", None, None, 0.0
        try:
            while final is None and failed is None:
                try:
                    kind, value = events.get(timeout=REDRAW_SECS)
                except queue.Empty:
                    kind, value = None, None
                if kind == "step":
                    node, step_status = value
                    mark = {"ok": "✓", "degraded": "⚠", "error": "✗"}[step_status]
                    status.write(f"{mark} {STEP_LABELS.get(node, node)}")
                    if node in REPLY_NODES:
                        status.update(label="Done", state="complete", expanded=False)
                elif kind == "token":
                    if not text:
                        status.update(label="Writing…", expanded=False)
                    text = value
                elif kind == "done":
                    final = value
                elif kind == "error":
                    failed = value
                elif kind == "notice":                 # e.g. "Google is busy, retrying in 30 s..."
                    status.update(label=value, state="running", expanded=True)
                    status.write(f"\u23f3 {value}")
                # an st.* call at least every REDRAW_SECS is what lets a Stop click interrupt this loop
                if kind is None or time.monotonic() - last_draw > REDRAW_SECS:
                    reply.markdown(to_streamlit_md(text) + " ▌")   # unfinished $..$ stays plain until closed
                    last_draw = time.monotonic()
        except BaseException:                # Stop click, another widget, or the tab was closed
            with saver.lock:
                stop.set()                   # from here on the worker can't write to the thread
                if pre_turn:                 # make the pre-turn checkpoint the latest again
                    graph.update_state(pre_turn, None, as_node="__copy__")
                else:
                    saver.delete_thread(st.session_state.thread)
            raise

        stop_slot.empty()
        if failed is not None:
            status.update(label="Failed", state="error")
            st.error(f"{type(failed).__name__}: {failed}")
            st.stop()
        reply.markdown(to_streamlit_md(final))
    st.rerun()                               # redraw history + sidebar from the saved state
