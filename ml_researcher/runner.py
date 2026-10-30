"""Run one chat turn through the graph with streaming + telemetry.

Logs one "step" event per graph node and one "request" event per turn, with:
latency, TTFT (time to first reply token), TPOT (time per output token after the first),
token usage, route, turn number and status / error_code.
"""
import logging
from collections import Counter
import time
import uuid
from typing import Callable, Optional

from langchain_core.messages import AIMessageChunk, HumanMessage

from .config import LOG_PROMPTS
from .llm import text_of

log = logging.getLogger("ml_researcher.requests")
REPLY_NODES = {"synthesizer", "answer"}             # only these nodes' tokens are the visible reply


class TurnCancelled(Exception):
    """Raised inside a turn once its should_stop() returns True."""


def _chunk_text(chunk) -> str:
    t = chunk.text
    return t if isinstance(t, str) else t()


def _error_code(e: Exception) -> str:
    code = getattr(e, "code", None)                  # google.genai APIError -> HTTP status
    return str(code) if isinstance(code, int) else type(e).__name__


def _ms(seconds: Optional[float]) -> Optional[int]:
    return None if seconds is None else round(seconds * 1000)


def run_turn(graph, thread_id: str, prompt: str,
             on_step: Optional[Callable[[str, str], None]] = None,
             on_token: Optional[Callable[[str], None]] = None,
             should_stop: Optional[Callable[[], bool]] = None) -> str:
    """Stream one turn. on_step(node, status) after each node; on_token(text_so_far) per reply chunk.
    should_stop() is polled on every stream event; True aborts the turn with TurnCancelled.
    Returns the final reply text; re-raises any error after logging it."""
    cfg = {"configurable": {"thread_id": thread_id}}
    ids = {"request_id": uuid.uuid4().hex[:12], "thread_id": thread_id}
    t0 = time.monotonic()
    started, steps = {}, []
    text, t_first, t_last = "", None, None
    reply_tokens, usage = 0, {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                                "reasoning_tokens": 0}
    status, error_code, error = "ok", None, None
    try:
        # "tasks" = node start/finish events, "messages" = (token chunk, metadata) from every LLM call
        for mode, data in graph.stream({"messages": [HumanMessage(prompt)]}, cfg,
                                       stream_mode=["tasks", "messages"]):
            if should_stop and should_stop():
                raise TurnCancelled()
            now = time.monotonic()
            if mode == "tasks":
                if "triggers" in data:               # task started
                    started[data["id"]] = now
                    continue
                result = data.get("result") or {}
                result = dict(result) if isinstance(result, list) else result
                step_status = "error" if data.get("error") else "ok"
                if step_status == "ok" and any(isinstance(v, str) and v.startswith("(no results")
                                               for v in result.values()):
                    step_status = "degraded"         # researcher caught a tool/API error
                step = {"node": data["name"], "status": step_status,
                        "duration_ms": _ms(now - started.pop(data["id"], now))}
                steps.append(step)
                log.info("step %s %s", data["name"], step_status, extra={"json_fields": {
                    **ids, "event": "step", **step,
                    "error": str(data["error"])[:300] if data.get("error") else None}})
                if on_step:
                    on_step(data["name"], step_status)
                continue

            chunk, meta = data
            if not isinstance(chunk, AIMessageChunk):  # skip full messages re-emitted from state
                continue
            u = chunk.usage_metadata or {}
            reasoning = (u.get("output_token_details") or {}).get("reasoning", 0)
            usage["input_tokens"] += u.get("input_tokens", 0)
            usage["cached_tokens"] += (u.get("input_token_details") or {}).get("cache_read", 0)   # explicit cache hits
            usage["output_tokens"] += u.get("output_tokens", 0)
            usage["reasoning_tokens"] += reasoning
            if meta.get("langgraph_node") not in REPLY_NODES:
                continue
            reply_tokens += u.get("output_tokens", 0) - reasoning   # visible tokens only
            piece = _chunk_text(chunk)
            if piece:
                t_first = t_first or now
                t_last = now
                text += piece
                if on_token:
                    on_token(text)
    except Exception as e:
        if isinstance(e, TurnCancelled) or (should_stop and should_stop()):
            status = "cancelled"
        else:
            status, error_code, error = "error", _error_code(e), str(e)[:500]
        raise
    finally:
        latency = time.monotonic() - t0
        values = graph.get_state(cfg).values
        turn = sum(isinstance(m, HumanMessage) for m in values.get("messages", []))
        tpot = (t_last - t_first) / (reply_tokens - 1) if t_first and reply_tokens > 1 else None
        # tools run this turn; an answer-only turn runs none (the state still holds the last research turn's list)
        calls = ((values.get("tool_calls") or []) + (["url_reader"] if values.get("read_url") else [])
                 if values.get("route") == "research" else [])
        fields = {**ids, "event": "request", "status": status, "error_code": error_code, "error": error,
                  "route": values.get("route"), "turn": turn,
                  "tool_calls": len(calls), "tool_call_counts": dict(Counter(calls)),
                  "latency_ms": _ms(latency),
                  "ttft_ms": _ms(t_first - t0 if t_first else None),
                  "tpot_ms": None if tpot is None else round(tpot * 1000, 2),
                  "reply_tokens": reply_tokens, **usage,
                  "steps": len(steps), "step_errors": sum(s["status"] != "ok" for s in steps),
                  "prompt_chars": len(prompt), "reply_chars": len(text)}
        if LOG_PROMPTS:
            fields["prompt"] = prompt[:500]
        log.log(logging.ERROR if status == "error" else logging.INFO,
                "request %s %s %sms", status, fields["route"], fields["latency_ms"],
                extra={"json_fields": fields})
    return text_of(values["messages"][-1])
