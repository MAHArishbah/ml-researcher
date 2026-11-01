"""Dashboard page (registered in ui.py): the local log dashboard. Refreshes every 30 s.

Complements the Cloud Monitoring dashboard rather than repeating it: Cloud shows p50/p95 and counts over time;
this page breaks things down per model, per turn and per conversation, and shows the SDK's retried calls,
which never reach Cloud Logging.
"""
import altair as alt
import pandas as pd
import streamlit as st

from ml_researcher import log_data
from ml_researcher.config import DASHBOARD_TZ

RANGES = {"Last 24 hours": pd.Timedelta(hours=24), "Last 7 days": pd.Timedelta(days=7), "All": None}
STATUS_COLORS = alt.Scale(domain=["ok", "degraded", "error", "unfinished"],
                          range=["#4c9a6a", "#d9a03f", "#d1495b", "#9a9a9a"])


@st.cache_data(show_spinner=False)
def data(signature: tuple) -> dict:          # re-read only when a log file changed
    return log_data.load()


def pcts(s: pd.Series) -> tuple:
    s = s.dropna()
    return (s.quantile(0.5), s.quantile(0.95)) if len(s) else (None, None)


def local(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert(DASHBOARD_TZ)


d = data(log_data.signature())
with st.sidebar:
    st.subheader("Filters")
    window = st.selectbox("Time range", list(RANGES), index=1, key="dash_range")
    routes = st.multiselect("Route", ["research", "answer"], default=["research", "answer"], key="dash_routes")
    all_models = sorted(d["requests"]["model"].unique()) if not d["requests"].empty else []
    models = st.multiselect("Main model", all_models, default=all_models, key="dash_models")
    st.caption(f"Times in {DASHBOARD_TZ} (set DASHBOARD_TZ in .env). Refreshes every 30 s.")


@st.fragment(run_every="30s")
def dashboard():
    d = data(log_data.signature())
    req, steps, retries = d["requests"], d["steps"], d["retries"]
    st.title("Agent dashboard")
    if req.empty:
        st.info(f"No requests logged yet in {log_data.LOG_FILE}.")
        return
    since = pd.Timestamp.now(tz="UTC") - RANGES[window] if RANGES[window] is not None else None
    if since is not None:
        req, retries = req[req["ts"] >= since], retries[retries["ts"] >= since]
    req = req[req["route"].isin(routes) & req["model"].isin(models)]
    st.caption(f"{len(req)} turns from the local log. The Cloud Monitoring dashboard has the p50/p95 trends; "
               "this page breaks them down per model, per turn and per conversation.")
    if req.empty:
        st.info("No turns match the filters.")
        return

    # ---- 1. speed by model ----
    st.header("1. Speed by model")
    rows = []
    for model, g in req.groupby("model"):
        ok = g[g["status"] == "ok"]
        (t50, t95), (l50, l95) = pcts(ok["ttft_s"]), pcts(ok["latency_s"])
        rows.append({"main model": model, "turns": len(g), "failed": (g["status"] == "error").mean(),
                     "first word p50 (s)": t50, "first word p95 (s)": t95,
                     "latency p50 (s)": l50, "latency p95 (s)": l95,
                     "cached input": g["cached_tokens"].sum() / max(g["input_tokens"].sum(), 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "failed": st.column_config.NumberColumn(format="percent"),
        "cached input": st.column_config.NumberColumn(format="percent"),
        **{c: st.column_config.NumberColumn(format="%.1f") for c in
           ("first word p50 (s)", "first word p95 (s)", "latency p50 (s)", "latency p95 (s)")}})
    dots = req[req["status"] == "ok"].dropna(subset=["ttft_s"]).assign(time=lambda x: local(x["ts"]))
    st.altair_chart(alt.Chart(dots).mark_circle(size=70, opacity=0.75).encode(
        x=alt.X("input_tokens:Q", title="input tokens in the turn"),
        y=alt.Y("ttft_s:Q", title="time to first word (s)"),
        color=alt.Color("model:N", title="main model"), shape=alt.Shape("route:N"),
        tooltip=["time:T", "thread_id:N", "route:N", "model:N", alt.Tooltip("ttft_s:Q", format=".1f"),
                 "input_tokens:Q"]).properties(height=280), width="stretch")
    st.caption("Each dot is one turn. A model whose dots sit high at every input size is slow to answer, "
               "not overloaded by context.")

    # ---- 2. where the time goes ----
    st.header("2. Where the time goes, per turn")
    recent = req.sort_values("ts").tail(15)
    spans = log_data.phase_spans(recent, steps).merge(
        recent[["request_id", "ts", "thread_id", "route", "status"]], on="request_id")
    spans["turn"] = (local(spans["ts"]).dt.strftime("%d %b %H:%M") + " · " + spans["thread_id"].str[:22]
                     + " · " + spans["route"] + ("" if spans.empty else spans["status"].map(
                         lambda s: "" if s == "ok" else f" ({s})")))
    order = spans.sort_values("ts", ascending=False)["turn"].unique().tolist()
    st.altair_chart(alt.Chart(spans).mark_bar().encode(
        y=alt.Y("turn:N", sort=order, title=None),
        x=alt.X("sum(seconds):Q", title="seconds (wall clock)"),
        color=alt.Color("phase:N", scale=alt.Scale(domain=log_data.PHASES), sort=log_data.PHASES),
        order=alt.Order("phase_rank:Q"),
        tooltip=["turn:N", "phase:N", alt.Tooltip("seconds:Q", format=".1f")]).transform_calculate(
        phase_rank=f"indexof({log_data.PHASES}, datum.phase)").properties(height=34 * max(len(order), 3)),
        width="stretch")
    st.caption("Researchers run in parallel, so 'research' is first start to last finish. "
               "'unfinished / waiting' is a step still running when the turn failed or was stopped.")

    # ---- 3. reliability ----
    st.header("3. Reliability: retried calls and failed turns")
    if retries.empty:
        st.success("No retried model calls in this range.")
    else:
        per_hour = retries.assign(hour=local(retries["ts"]).dt.floor("h"))
        st.altair_chart(alt.Chart(per_hour).mark_bar().encode(
            x=alt.X("hour:T", title=None), y=alt.Y("count():Q", title="retried calls"),
            color=alt.Color("cause:N", title="cause"), tooltip=["hour:T", "cause:N", "count():Q"]
        ).properties(height=220), width="stretch")
        st.caption("ReadTimeout = the model sent nothing for the whole silence limit; ConnectError = no "
                   "connection; 429 busy = Google refused for capacity. Each costs a retry and its wait.")
    bad = req[req["status"] != "ok"].sort_values("ts", ascending=False)
    if bad.empty:
        st.success("No failed or stopped turns in this range.")
    else:
        failed_step = (steps[steps["status"] == "error"].groupby("request_id")["node"].first()
                       if not steps.empty else pd.Series(dtype=str))
        st.dataframe(pd.DataFrame({
            "time": local(bad["ts"]).dt.strftime("%d %b %H:%M:%S"), "conversation": bad["thread_id"],
            "route": bad["route"], "model": bad["model"], "status": bad["status"],
            "failed step": bad["request_id"].map(failed_step).fillna("—"),
            "latency (s)": bad["latency_s"].round(0), "error": bad["error"].fillna("").str[:300]}),
            hide_index=True, width="stretch", column_config={      # fixed widths: wider than the page -> scrolls
                "time": st.column_config.TextColumn(width=130), "conversation": st.column_config.TextColumn(width=170),
                "route": st.column_config.TextColumn(width=80), "model": st.column_config.TextColumn(width=140),
                "status": st.column_config.TextColumn(width=85), "failed step": st.column_config.TextColumn(width=110),
                "latency (s)": st.column_config.NumberColumn(width=85, format="%d"),
                "error": st.column_config.TextColumn(width=900, help="Click a cell to see its full text")})
        st.caption("Scroll the table sideways for the full error, or click an error cell to open it.")

    # ---- 4. context growth ----
    st.header("4. Context growth per conversation")
    active = req.sort_values("ts").groupby("thread_id")["ts"].max().nlargest(8).index
    # completed turns only: a failed / stopped turn is rolled back and its retry reuses the turn number
    growth = (req[req["thread_id"].isin(active) & (req["status"] == "ok")].sort_values("ts")
              .drop_duplicates(["thread_id", "turn"], keep="last").assign(time=lambda x: local(x["ts"])))
    zoom = alt.selection_interval(bind="scales", encodings=["x"])   # scroll = zoom the turn axis, drag = pan
    st.altair_chart(alt.Chart(growth).mark_line(point=True).add_params(zoom).encode(
        x=alt.X("turn:Q", title="turn in the conversation", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("input_tokens:Q", title="input tokens"),
        color=alt.Color("thread_id:N", title="conversation"),
        tooltip=["thread_id:N", "turn:Q", "route:N", "input_tokens:Q", "cached_tokens:Q", "time:T"]
    ).properties(height=280), width="stretch")
    st.caption("The 8 most recently active conversations, completed turns only. Research turns re-read sources and "
               "jump up; answer turns should stay flat once older turns are summarised. Scroll on the chart to zoom "
               "the turn axis, drag to pan, double-click to reset.")

    # ---- 5. tool mix ----
    st.header("5. Tool mix (research turns)")
    tools = log_data.tool_counts(req)
    if tools.empty:
        st.info("No research turns in this range.")
    else:
        st.altair_chart(alt.Chart(tools).mark_bar().encode(
            y=alt.Y("tool:N", sort="-x", title=None), x=alt.X("per_turn:Q", title="calls per research turn"),
            tooltip=["tool:N", "calls:Q", alt.Tooltip("per_turn:Q", format=".2f"),
                     alt.Tooltip("turns_using:Q", format=".0%", title="share of turns using it")]
        ).properties(height=30 * len(tools) + 20), width="stretch")

    # ---- 6. one turn's timeline ----
    st.header("6. One turn's timeline")
    latest = req.sort_values("ts", ascending=False).head(50)
    labels = {r.request_id: f"{local(pd.Series([r.ts])).iloc[0]:%d %b %H:%M:%S} · {r.thread_id[:24]} · "
                            f"{r.route} · {r.latency_s:.0f} s · {r.status}" for r in latest.itertuples()}
    rid = st.selectbox("Turn", list(labels), format_func=labels.get, key="dash_turn")
    r = latest.set_index("request_id").loc[rid]
    s = steps[steps["request_id"] == rid].copy() if not steps.empty else pd.DataFrame()
    tool_calls = r.get("tool_calls")
    st.caption(f"main model **{r['model']}** · status {r['status']} · "
               + (f"first word {r['ttft_s']:.1f} s" if pd.notna(r["ttft_s"]) else "no reply text")
               + f" · input {int(r['input_tokens']):,} tokens"
               + (f" · {int(tool_calls)} tool calls" if pd.notna(tool_calls) else ""))
    bars = []
    for row in s.itertuples():
        bars.append({"node": row.node, "status": row.status,
                     "from": (row.start - r["start"]).total_seconds(), "to": (row.end - r["start"]).total_seconds()})
    last_end = max((b["to"] for b in bars), default=0)
    if r["latency_s"] - last_end > 1:                     # time after the last finished step: a stuck call
        bars.append({"node": "(no step finished)", "status": "unfinished", "from": last_end, "to": r["latency_s"]})
    gantt = pd.DataFrame(bars)
    in_turn = retries[(retries["ts"] >= r["start"]) & (retries["ts"] <= r["ts"])]
    marks = in_turn.assign(at=(in_turn["ts"] - r["start"]).dt.total_seconds())
    order = list(dict.fromkeys(gantt.sort_values("from")["node"])) if not gantt.empty else None  # by first start
    chart = alt.Chart(gantt).mark_bar(height=16).encode(
        y=alt.Y("node:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelPadding=6)),
        x=alt.X("from:Q", title="seconds since the turn started"), x2="to:Q",
        color=alt.Color("status:N", scale=STATUS_COLORS, legend=alt.Legend(orient="bottom", direction="horizontal")),
        tooltip=["node:N", "status:N", alt.Tooltip("from:Q", format=".1f"), alt.Tooltip("to:Q", format=".1f")])
    if not marks.empty:
        chart += alt.Chart(marks).mark_rule(color="#d1495b", strokeDash=[4, 3], size=2).encode(
            x="at:Q", tooltip=["cause:N", alt.Tooltip("at:Q", format=".1f", title="retried at (s)")])
    # height per row (not a fixed total), so 2 rows never squeeze together and 12 rows never get cramped
    st.altair_chart(chart.properties(height=alt.Step(30)), width="stretch")
    st.caption("Red dashed lines are retried model calls during this turn. A long grey bar is time spent "
               "waiting on a call that never finished.")


dashboard()
