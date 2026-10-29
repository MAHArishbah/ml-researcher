"""Summarise the local request log:  python -m ml_researcher.log_report [--last 200]"""
import argparse
import json
import math
from collections import Counter, defaultdict

from .logging_config import LOG_FILE


def pct(values: list, p: float):
    """Nearest-rank percentile (p in 0..100); None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def load(last: int) -> tuple:
    requests, steps = [], []
    files = sorted(LOG_FILE.parent.glob(LOG_FILE.name + "*"), reverse=True)  # rotated files first (oldest)
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            (requests if e.get("event") == "request" else steps if e.get("event") == "step" else []).append(e)
    return requests[-last:], steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=500, help="only the most recent N requests")
    requests, steps = load(ap.parse_args().last)
    if not requests:
        print(f"no requests logged yet in {LOG_FILE}")
        return
    ids = {r["request_id"] for r in requests}
    print(f"{len(requests)} requests | status {dict(Counter(r['status'] for r in requests))} "
          f"| errors {dict(Counter(r['error_code'] for r in requests if r['error_code']))} "
          f"| routes {dict(Counter(r['route'] for r in requests))}")
    print(f"\n{'metric':<22}{'p50':>10}{'p95':>10}{'max':>10}")
    for key, label in [("latency_ms", "latency (ms)"), ("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms/token)"),
                       ("reply_tokens", "reply tokens"), ("output_tokens", "output tokens (all)"),
                       ("input_tokens", "input tokens"), ("cached_tokens", "cached input tokens"),
                       ("turn", "turn number")]:
        vals = [r[key] for r in requests if r.get(key) is not None]
        p50, p95 = pct(vals, 50), pct(vals, 95)
        print(f"{label:<22}{'-' if p50 is None else p50:>10}{'-' if p95 is None else p95:>10}"
              f"{max(vals, default='-'):>10}")

    by_node = defaultdict(list)
    for s in steps:
        if s.get("request_id") in ids:
            by_node[s["node"]].append(s)
    print(f"\n{'step':<22}{'p50 ms':>10}{'p95 ms':>10}{'n':>6}{'not ok':>8}")
    for node, ss in sorted(by_node.items()):
        d = [s["duration_ms"] for s in ss]
        print(f"{node:<22}{pct(d, 50):>10}{pct(d, 95):>10}{len(ss):>6}{sum(s['status'] != 'ok' for s in ss):>8}")


if __name__ == "__main__":
    main()
