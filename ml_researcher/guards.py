"""Loop guards: stable hashes for tool calls, tool results and follow-up actions.

Used to detect repeats (cycle detection) and lack of progress (identical results) in
  - each researcher's tool loop (graph.make_researcher), and
  - the reviewer -> follow-up cycle (graph.reviewer).
"""
import hashlib
import json
import re

_WORD = re.compile(r"[a-z0-9]+")


def digest(obj) -> str:
    """Short stable hash of any JSON-able value (dict key order doesn't matter)."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def normalize_query(text: str) -> str:
    """Case, punctuation and word order don't make a query new: 'Flow matching, ODE' == 'ode flow-matching'."""
    return " ".join(sorted(set(_WORD.findall(str(text).lower()))))


def call_key(tool: str, args: dict) -> str:
    """Identity of a tool call, with string arguments normalised."""
    return digest([tool, {k: normalize_query(v) if isinstance(v, str) else v for k, v in args.items()}])


def action_key(source: str, query: str) -> str:
    """Identity of a follow-up action (source + normalised query)."""
    return digest([source, normalize_query(query)])


def outcome(result) -> str:
    """One-line summary of a tool result, for the attempt history shown back to the model."""
    if not isinstance(result, dict):
        return f"returned {str(result)[:80]}"
    if result.get("status") == "error":
        return f"error: {str(result.get('message'))[:150]}"
    lists = [v for v in result.values() if isinstance(v, list)]
    return f"{len(lists[0])} results" if lists else "ok"


def is_empty(result) -> bool:
    """Error or zero results: nothing usable came back."""
    return isinstance(result, dict) and (result.get("status") == "error" or outcome(result) == "0 results")
