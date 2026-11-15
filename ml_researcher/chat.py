"""Terminal chat with persistent threads:  python -m ml_researcher.chat --thread flow-matching"""
import argparse
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from . import memory
from .config import CHECKPOINT_DB, private_dir
from .graph import build_graph
from .logging_config import setup_logging
from .runner import run_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", default="default", help="conversation id; reuse it to continue later")
    args = ap.parse_args()
    setup_logging(app="chat")

    private_dir(CHECKPOINT_DB.parent)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": args.thread}}
    print(f"thread '{args.thread}'  |  commands: /state  /graph  /remember <fact>  /quit  (add /memory to a message)")
    try:
        while True:
            try:
                text = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/graph":
                print(graph.get_graph().draw_mermaid())
                continue
            if text == "/state":
                vals = graph.get_state(config).values
                print({k: (v if k in ("route", "topic", "search_plan") else f"<{len(str(v))} chars>")
                    for k, v in vals.items()})
                continue
            if (note := memory.command(text)) is not None:      # /remember: stored, not sent as a turn
                print(f"  {note}")
                continue
            printed, shown = 0, ""

            def on_token(so_far: str):
                nonlocal printed, shown
                if not so_far.startswith(shown):          # a retried reply call: the text starts over
                    print("\n  (retried: the reply starts over)")
                    printed = 0
                if not printed:
                    print("\nassistant>")
                print(so_far[printed:], end="", flush=True)   # print only the new part
                printed, shown = len(so_far), so_far

            try:
                run_turn(graph, args.thread, text,
                         on_step=lambda node, status: print(f"  {'✓' if status == 'ok' else '⚠'} {node}"),
                         on_token=on_token,
                         on_notice=lambda message: print(f"  \u23f3 {message}"))
                print()
            except KeyboardInterrupt:
                print("\n  (stopped)")
            except Exception as e:
                print(f"\n  error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()