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


# ---------- prompt-injection guards ----------
# Tool output (abstracts, repo descriptions, video descriptions, the user's own files) is untrusted text.
# The tools can only search, so the main risks are misleading content and planted links:
#   - links: a brief may only contain URLs that a tool returned in its structured "url" field;
#   - instruction-like phrases in tool output are flagged (logged + a note to the model), never obeyed.

_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+")
_SUSPICIOUS = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|messages)"
    r"|disregard (all |the )?(previous|prior|above|system)"
    r"|(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)"
    r"|\bsystem prompt\b|\bnew instructions?\s*:|\byou are now\b|\bjailbreak\b|\bdo anything now\b"
    r"|</?\s*(system|assistant|instructions?)\s*>",
    re.IGNORECASE)


def source_urls(result) -> list:
    """URLs a tool returned in its structured "url" fields (built from API data, not from free text)."""
    found = []
    if isinstance(result, dict):
        for k, v in result.items():
            found += [v] if k == "url" and isinstance(v, str) else source_urls(v)
    elif isinstance(result, list):
        for v in result:
            found += source_urls(v)
    return found


def norm_url(url: str) -> str:
    """Compare links loosely: scheme, www., trailing slash, arXiv version and abs/pdf don't matter."""
    u = re.sub(r"^https?://(www\.)?", "", url.strip().rstrip(".,;:!?)")).lower().rstrip("/")
    u = re.sub(r"^arxiv\.org/(abs|pdf)/([^/?#]+?)(v\d+)?(\.pdf)?$", r"arxiv.org/abs/\2", u)
    return re.sub(r"^youtu\.be/([\w-]+)", r"youtube.com/watch?v=\1", u)


def filter_urls(text: str, allowed) -> tuple[str, list]:
    """Remove links that aren't in `allowed` (or below one, e.g. a file inside an allowed repo).
    [label](bad) keeps only the label; a bare bad URL becomes "[link removed]". Returns (text, removed)."""
    ok_set = {norm_url(u) for u in allowed}
    removed = []

    def ok(url):
        n = norm_url(url)
        return n in ok_set or any(n.startswith(a + "/") for a in ok_set)

    def md(m):
        if ok(m[2]):
            return m[0]
        removed.append(m[2])
        return m[1]

    def bare(m):
        url = m[0].rstrip(".,;:!?")
        if ok(url):
            return m[0]
        removed.append(url)
        return "[link removed]" + m[0][len(url):]

    return _BARE_URL.sub(bare, _MD_LINK.sub(md, text)), removed


def suspicious(text: str) -> list:
    """Instruction-like phrases found in untrusted text (distinct, lower-cased)."""
    return sorted({m[0].lower() for m in _SUSPICIOUS.finditer(text or "")})
