"""Ingestion as a LangGraph map-reduce:
discover -> plan -> index_file (one Send per new/changed file, parallel) -> cleanup.

Run from /mnt/e/AI_ML/Prac/researchAgent:
    python -m ml_researcher.ingest /mnt/e/AI_ML/Books "E:\\AI_ML\\Notes"
    python -m ml_researcher.ingest            # no argument -> LOCAL_DOCS_DIR from .env
"""
import argparse
import operator
import os
from collections import Counter
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tqdm import tqdm
import time
import logging

from . import vectordb
from .config import DOCS_DIR, FILE_WORKERS, VECTOR_BACKEND
from .readers import to_wsl_path, walk
from .logging_config import setup_logging

log = logging.getLogger("ml_researcher.ingest")


class IngestState(TypedDict,total=False):
    roots: list                                   # folders to index (Linux paths)
    files: list                                   # [{"path", "mtime"}] found on disk
    todo: list                                    # new or changed files
    skipped: int                                  # unchanged files
    gone: list                                    # indexed paths deleted from disk
    results: Annotated[list, operator.add]        # reducer: parallel workers append here
    removed: int

class FileTask(TypedDict):                        # payload of each Send
    path: str
    mtime: float

def discover(state: IngestState) -> dict:
    files = []
    for root in state["roots"]:
        for p in walk(Path(root)):
            try:
                files.append({"path": str(p), "mtime": p.stat().st_mtime})
            except OSError:
                pass
    return {"files": files}

def plan(state: IngestState) -> dict:
    known = vectordb.indexed_files()
    todo = [f for f in state["files"] if known.get(f["path"]) != f["mtime"]]
    todo.sort(key=lambda f: os.path.getsize(f["path"])) #index the shorter files first
    on_disk = {f["path"] for f in state["files"]}
    prefixes = tuple(r.rstrip(os.sep) + os.sep for r in state["roots"])
    gone = [p for p in known if p.startswith(prefixes) and p not in on_disk]
    return {"todo": todo, "skipped": len(state["files"]) - len(todo), "gone": gone}

def fan_out(state: IngestState):
    if not state["todo"]:
        return "cleanup"
    return [Send("index_file", {"path": f["path"], "mtime": f["mtime"]}) for f in state["todo"]]

def index_file(task: FileTask) -> dict:
    name, t0 = Path(task["path"]).name, time.monotonic()
    tqdm.write(f"  -> {name}")
    try:
        n = vectordb.upsert_file(Path(task["path"]), task["mtime"])
        res = {"path": task["path"], "status": "ok" if n else "empty", "chunks": n}
    except Exception as e:
        res = {"path": task["path"], "status": "failed", "error": str(e)[:200]}
    log.info("ingest %s %s", res["status"], name, extra={"json_fields": {
        "event": "ingest_file", **res, "duration_ms": round((time.monotonic() - t0) * 1000)}})
    return {"results": [res]}

def cleanup(state: IngestState) -> dict:
    gone = state.get("gone", [])
    vectordb.delete_paths(gone)
    return {"removed": len(gone)}

def build_ingest_graph():
    g = StateGraph(IngestState)
    g.add_node("discover", discover)
    g.add_node("plan", plan)
    g.add_node("index_file", index_file)
    g.add_node("cleanup", cleanup)
    g.add_edge(START, "discover")
    g.add_edge("discover", "plan")
    g.add_conditional_edges("plan", fan_out, ["index_file", "cleanup"])
    g.add_edge("index_file", "cleanup")           # runs once, after all Sends finish
    g.add_edge("cleanup", END)
    return g.compile()

def main():
    setup_logging(app="ingest")
    ap = argparse.ArgumentParser(description="Recursively index PDF/Word/Markdown files.")
    ap.add_argument("folders", nargs="*", help="folders to index (Linux or Windows paths)")
    args = ap.parse_args()

    roots = []
    for raw in args.folders or [str(DOCS_DIR)]:
        r = to_wsl_path(raw)
        if r.is_dir():
            roots.append(str(r))
        else:
            print(f"not a folder: {r}")
    if not roots:
        return
    print(f"backend: {VECTOR_BACKEND} | roots: {roots}")
    graph, bar, results = build_ingest_graph(), None, []
    for update in graph.stream({"roots": roots}, {"max_concurrency": FILE_WORKERS},
                               stream_mode="updates"):
        for node, out in update.items():
            if node == "discover":
                counts = Counter(Path(f["path"]).suffix.lower() for f in out["files"])
                print(f"found {len(out['files'])} files {dict(counts)}")
            elif node == "plan":
                print(f"to index {len(out['todo'])} | unchanged {out['skipped']} | "
                      f"deleted on disk {len(out['gone'])}")
                bar = tqdm(total=len(out["todo"]), desc="indexing", dynamic_ncols=True)
            elif node == "index_file":
                r = out["results"][0]
                results.append(r)
                bar.update(1)
                if r["status"] == "failed":
                    tqdm.write(f"  failed {Path(r['path']).name}: {r['error']}")
            elif node == "cleanup":
                if bar:
                    bar.close()
                print(f"removed {out['removed']} deleted files")

    s = Counter(r["status"] for r in results)
    print(f"done: ok {s['ok']}, no text {s['empty']}, failed {s['failed']} | "
          f"items in DB: {vectordb.count()}")
if __name__ == "__main__":
    main()