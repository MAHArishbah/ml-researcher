"""Create/update log-based metrics + a Cloud Monitoring dashboard for the research agent.

    python scripts/setup_gcp_metrics.py

Reads the structured "request"/"step" log entries written by ml_researcher.runner (LOG_TO_GCP=true)
and turns them into metrics. p50/p95 are computed by Cloud Monitoring from the distribution metrics.
Safe to re-run: metrics are upserted, the dashboard is replaced.
"""
import sys
from pathlib import Path

import google.auth
from google.auth.transport.requests import AuthorizedSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_researcher.config import LOG_NAME, PROJECT  # noqa: E402

LOGGING = f"https://logging.googleapis.com/v2/projects/{PROJECT}/metrics"
DASHBOARDS = f"https://monitoring.googleapis.com/v1/projects/{PROJECT}/dashboards"
DASHBOARD_NAME = "ML Researcher Agent"
PREFIX = "ml_researcher_"
BASE = f'logName="projects/{PROJECT}/logs/{LOG_NAME}"'
REQ = f'{BASE} AND jsonPayload.event="request"'
STEP = f'{BASE} AND jsonPayload.event="step"'


def exp_buckets(scale: float, growth: float = 1.15, n: int = 128) -> dict:
    """~15% wide buckets -> percentiles accurate to roughly +-7%."""
    return {"exponentialBuckets": {"numFiniteBuckets": n, "growthFactor": growth, "scale": scale}}


def counter(name, desc, filt, labels):
    return {"name": PREFIX + name, "description": desc, "filter": filt,
            "metricDescriptor": {"metricKind": "DELTA", "valueType": "INT64", "unit": "1",
                                 "labels": [{"key": k, "valueType": "STRING"} for k in labels]},
            "labelExtractors": {k: f"EXTRACT(jsonPayload.{k})" for k in labels}}


def distribution(name, desc, filt, field, unit, buckets, labels):
    m = counter(name, desc, f"{filt} AND jsonPayload.{field}>=0", labels)
    m["metricDescriptor"].update(valueType="DISTRIBUTION", unit=unit)
    m.update(valueExtractor=f"EXTRACT(jsonPayload.{field})", bucketOptions=buckets)
    return m


METRICS = [
    counter("requests", "Chat turns by status / error code / route", REQ, ["status", "error_code", "route"]),
    distribution("request_latency", "End-to-end turn latency", REQ, "latency_ms", "ms",
                 exp_buckets(10), ["route", "status"]),
    distribution("ttft", "Time to first reply token", REQ, "ttft_ms", "ms", exp_buckets(10), ["route"]),
    distribution("tpot", "Time per output token after the first", REQ, "tpot_ms", "ms",
                 exp_buckets(0.1, n=100), ["route"]),
    distribution("turns", "Turn number within a conversation", REQ, "turn", "1",
                 {"linearBuckets": {"numFiniteBuckets": 50, "width": 1, "offset": 0}}, ["route"]),
    distribution("output_tokens", "Output tokens per turn (all LLM calls, incl. reasoning)", REQ,
                 "output_tokens", "1", exp_buckets(1, 1.3, 64), ["route"]),
    distribution("input_tokens", "Input (prompt) tokens per turn, all LLM calls", REQ,
                 "input_tokens", "1", exp_buckets(1, 1.3, 64), ["route"]),
    distribution("reply_tokens", "Visible reply tokens per turn", REQ, "reply_tokens", "1",
                 exp_buckets(1, 1.3, 64), ["route"]),
    counter("steps", "Graph node executions by node / status", STEP, ["node", "status"]),
    distribution("step_latency", "Graph node duration", STEP, "duration_ms", "ms",
                 exp_buckets(10), ["node", "status"]),
]


def series(metric, reducer, legend, group_by=(), extra_filter="", aligner="ALIGN_DELTA"):
    agg = {"alignmentPeriod": "300s", "perSeriesAligner": aligner, "crossSeriesReducer": reducer}
    if group_by:
        agg["groupByFields"] = list(group_by)
    return {"plotType": "LINE", "legendTemplate": legend, "minAlignmentPeriod": "300s",
            "timeSeriesQuery": {"timeSeriesFilter": {
                "filter": f'metric.type="logging.googleapis.com/user/{PREFIX}{metric}"{extra_filter}',
                "aggregation": agg}}}


def p50_p95(title, metric, unit_label):
    return {"title": title, "xyChart": {
        "dataSets": [series(metric, "REDUCE_PERCENTILE_50", "p50"),
                     series(metric, "REDUCE_PERCENTILE_95", "p95")],
        "yAxis": {"label": unit_label, "scale": "LINEAR"}}}


def dashboard() -> dict:
    widgets = [
        {"title": "Requests by status (per 5 min)", "xyChart": {"dataSets": [
            {**series("requests", "REDUCE_SUM", "${metric.labels.status}", ["metric.label.status"]),
             "plotType": "STACKED_BAR"}]}},
        {"title": "Errors by code (per 5 min)", "xyChart": {"dataSets": [
            {**series("requests", "REDUCE_SUM", "${metric.labels.error_code}", ["metric.label.error_code"],
                      ' AND metric.label.status="error"'), "plotType": "STACKED_BAR"}]}},
        p50_p95("Request latency p50 / p95", "request_latency", "ms"),
        p50_p95("TTFT p50 / p95", "ttft", "ms"),
        p50_p95("TPOT p50 / p95", "tpot", "ms / token"),
        {"title": "Step latency p95 by node", "xyChart": {"dataSets": [
            series("step_latency", "REDUCE_PERCENTILE_95", "${metric.labels.node}", ["metric.label.node"])],
            "yAxis": {"label": "ms", "scale": "LINEAR"}}},
        {"title": "Degraded / failed steps by node", "xyChart": {"dataSets": [
            {**series("steps", "REDUCE_SUM", "${metric.labels.node}", ["metric.label.node"],
                      ' AND metric.label.status!="ok"'), "plotType": "STACKED_BAR"}]}},
        p50_p95("Turn number p50 / p95", "turns", "turn"),
        p50_p95("Output tokens per turn p50 / p95", "output_tokens", "tokens"),
        p50_p95("Input tokens per turn p50 / p95", "input_tokens", "tokens"),
        {"title": "Requests by route (per 5 min)", "xyChart": {"dataSets": [
            {**series("requests", "REDUCE_SUM", "${metric.labels.route}", ["metric.label.route"]),
             "plotType": "STACKED_BAR"}]}},
    ]
    tiles = [{"xPos": (i % 2) * 6, "yPos": (i // 2) * 4, "width": 6, "height": 4, "widget": w}
             for i, w in enumerate(widgets)]
    return {"displayName": DASHBOARD_NAME, "mosaicLayout": {"columns": 12, "tiles": tiles}}


def check(resp, what):
    if resp.status_code >= 300:
        msg = resp.json().get("error", {}).get("message", resp.text) if resp.content else resp.reason
        sys.exit(f"{what} failed ({resp.status_code}): {msg}")


def main():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    s = AuthorizedSession(creds)
    print(f"project {PROJECT} | log {LOG_NAME}")
    for m in METRICS:
        check(s.put(f"{LOGGING}/{m['name']}", json=m), f"metric {m['name']}")   # PUT = create or update
        print(f"  metric  logging.googleapis.com/user/{m['name']}")

    r = s.get(DASHBOARDS)
    check(r, "list dashboards")
    for d in r.json().get("dashboards", []):
        if d.get("displayName") == DASHBOARD_NAME:
            check(s.delete(f"https://monitoring.googleapis.com/v1/{d['name']}"), "delete old dashboard")
    r = s.post(DASHBOARDS, json=dashboard())
    check(r, "create dashboard")
    dash_id = r.json()["name"].rsplit("/", 1)[-1]
    print(f"  dashboard https://console.cloud.google.com/monitoring/dashboards/builder/{dash_id}?project={PROJECT}")


if __name__ == "__main__":
    main()
