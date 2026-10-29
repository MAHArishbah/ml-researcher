"""Terminal chat with persistent threads:  python -m ml_researcher.chat --thread flow-matching"""
import argparse
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import CHECKPOINT_DB
from .graph import build_graph
from .logging_config import setup_logging
from .runner import run_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", default="default", help="conversation id; reuse it to continue later")
    args = ap.parse_args()
    setup_logging(app="chat")

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": args.thread}}
    print(f"thread '{args.thread}'  |  commands: /state  /graph  /quit")
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
            printed = 0

            def on_token(so_far: str):
                nonlocal printed
                if not printed:
                    print("\nassistant>")
                print(so_far[printed:], end="", flush=True)   # print only the new part
                printed = len(so_far)

            try:
                run_turn(graph, args.thread, text,
                         on_step=lambda node, status: print(f"  {'✓' if status == 'ok' else '⚠'} {node}"),
                         on_token=on_token)
                print()
            except KeyboardInterrupt:
                print("\n  (stopped)")
            except Exception as e:
                print(f"\n  error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()