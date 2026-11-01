"""The local JSON-lines log as DataFrames, for the dashboard page (app_pages/dashboard.py).

requests: one row per turn (event "request");  steps: one row per graph node run, with start/end;
retries: model calls that were retried (SDK "Retrying ... raised X" lines and the app's 429 back-offs).
"""
import json
import re

import pandas as pd

from .logging_config import LOG_FILE

RESEARCHERS = {"arxiv_researcher", "notes_researcher", "code_researcher", "github_researcher",
               "youtube_researcher", "web_researcher"}
PHASE_OF = {"router": "route", "planner": "plan", **dict.fromkeys(RESEARCHERS, "research"),
            "reviewer": "review", "followup": "review", "python_check": "review",
            "synthesizer": "write", "answer": "write"}
PHASES = ["route", "plan", "research", "review", "write", "unfinished / waiting"]
_RETRY = re.compile(r"raised (\w+)")


def log_files() -> list:
    """Current log plus rotated ones, oldest first (agent.jsonl.5 ... agent.jsonl.1, agent.jsonl)."""
    return sorted(LOG_FILE.parent.glob(LOG_FILE.name + "*"), reverse=True)


def signature() -> tuple:
    """Changes whenever a log file changes: the cache key for load()."""
    return tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in log_files())


def load() -> dict:
    requests, steps, retries = [], [], []
    for f in log_files():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = e.get("event")
            if event == "request":
                requests.append(e)
            elif event == "step":
                steps.append(e)
            elif event == "model_backoff":
                retries.append({"ts": e["ts"], "cause": "429 busy"})
            elif e.get("logger") == "google_genai._api_client" and (m := _RETRY.search(e.get("message", ""))):
                retries.append({"ts": e["ts"], "cause": m.group(1)})

    req = pd.DataFrame(requests)
    if not req.empty:
        req = req.drop_duplicates("request_id", keep="last")
        req["ts"] = pd.to_datetime(req["ts"], utc=True)
        req["model"] = req["main_model"].fillna("not logged") if "main_model" in req else "not logged"
        req["latency_s"] = req["latency_ms"] / 1000
        req["ttft_s"] = req["ttft_ms"] / 1000
        req["start"] = req["ts"] - pd.to_timedelta(req["latency_ms"].fillna(0), unit="ms")
    stp = pd.DataFrame(steps)
    if not stp.empty:
        stp = stp.drop_duplicates(["request_id", "node", "ts"])
        stp["end"] = pd.to_datetime(stp["ts"], utc=True)
        stp["start"] = stp["end"] - pd.to_timedelta(stp["duration_ms"].fillna(0), unit="ms")
        stp["phase"] = stp["node"].map(PHASE_OF).fillna(PHASES[-1])
    rt = pd.DataFrame(retries, columns=["ts", "cause"])
    rt["ts"] = pd.to_datetime(rt["ts"], utc=True)
    return {"requests": req, "steps": stp, "retries": rt}


def phase_spans(req: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock seconds per phase for each request (parallel researchers count once: first start to last
    end), plus "unfinished / waiting" = the rest of the latency: a step still running when the turn failed or was
    stopped, or time between steps. One row per (request_id, phase)."""
    if req.empty or steps.empty:
        return pd.DataFrame(columns=["request_id", "phase", "seconds"])
    s = steps[steps["request_id"].isin(req["request_id"])]
    spans = (s.groupby(["request_id", "phase"])
             .agg(first=("start", "min"), last=("end", "max")).reset_index())
    spans["seconds"] = (spans["last"] - spans["first"]).dt.total_seconds()
    other = (req.set_index("request_id")["latency_s"]
             - spans.groupby("request_id")["seconds"].sum()).clip(lower=0).dropna()
    rest = other.rename("seconds").reset_index().assign(phase=PHASES[-1])
    return pd.concat([spans[["request_id", "phase", "seconds"]], rest], ignore_index=True)


def tool_counts(req: pd.DataFrame) -> pd.DataFrame:
    """Calls per tool over research turns: total, per research turn, and share of turns that used it."""
    research = req[req["route"] == "research"] if not req.empty else req
    rows = [(tool, n) for counts in research.get("tool_call_counts", pd.Series(dtype=object)).dropna()
            for tool, n in counts.items()]
    if not rows:
        return pd.DataFrame(columns=["tool", "calls", "per_turn", "turns_using"])
    df = pd.DataFrame(rows, columns=["tool", "n"])
    out = df.groupby("tool").agg(calls=("n", "sum"), turns=("n", "size")).reset_index()
    out["per_turn"] = out["calls"] / len(research)
    out["turns_using"] = out["turns"] / len(research)
    return out.sort_values("calls", ascending=False)
