"""Optional web search for brand-new papers, behind strict guardrails.

Runs only when the user's message starts with /web (and WEB_SEARCH is on). Uses Gemini's built-in Google
Search grounding, but treats it purely as a source finder:
  1. Gemini's own grounded answer text is thrown away; only its list of sources is used.
  2. Each source is a Google redirect link; its real target is read from the redirect WITHOUT following it
     (the page itself is never fetched), and must be https on an allow-listed domain.
  3. arXiv papers are re-fetched from the official arXiv API: title, authors, date and abstract come from
     arXiv, not from the web page. Other allow-listed sites (OpenReview, GitHub, ...) are passed as links only.
  4. At most WEB_MAX_SOURCES results; only their verified URLs may appear in the brief.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import arxiv
import requests
from langchain_core.messages import HumanMessage, SystemMessage

from .config import MAIN_MODEL, WEB_SEARCH
from .prompts import prompt

log = logging.getLogger("ml_researcher.websearch")
WEB_MAX_SOURCES = 8
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?")
LINK_ONLY_DOMAINS = {"openreview.net", "github.com", "huggingface.co", "proceedings.neurips.cc",
                     "proceedings.mlr.press", "aclanthology.org", "openaccess.thecvf.com", "jmlr.org"}
SEARCH_PROMPT = prompt("WEB_SEARCH")
_llm = None


def available() -> bool:
    """Grounded search is a Gemini feature: on only for a google_genai MAIN_MODEL and WEB_SEARCH=true."""
    return WEB_SEARCH and MAIN_MODEL.startswith("google_genai:")


def _grounded_llm():
    global _llm
    if _llm is None:
        from .llm import chat_model
        _llm = chat_model(MAIN_MODEL, effort="low").bind_tools([{"google_search": {}}])
    return _llm


def _resolve(redirect: str) -> str | None:
    """Real target of a grounding redirect, read from its Location header (the target is never fetched)."""
    try:
        r = requests.get(redirect, allow_redirects=False, timeout=10)
        return r.headers.get("Location") if r.status_code in (301, 302, 303, 307, 308) else None
    except requests.RequestException:
        return None


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _allowed(url: str, domains: set) -> bool:
    host = _host(url)
    return urlparse(url).scheme == "https" and any(host == d or host.endswith("." + d) for d in domains)


def search(topic: str) -> tuple[str, list, dict]:
    """(findings text, verified URLs, stats). Findings contain only arXiv-API metadata and bare links."""
    reply = _grounded_llm().invoke([SystemMessage(SEARCH_PROMPT), HumanMessage(f"Topic: {topic}")])
    meta = reply.response_metadata.get("grounding_metadata") or {}
    chunks = [c["web"] for c in meta.get("grounding_chunks") or [] if c.get("web")]
    arxiv_ids, links, dropped = [], [], []
    candidates = chunks[: WEB_MAX_SOURCES * 2]
    with ThreadPoolExecutor(4) as ex:        # plain redirect lookups (not model calls), independent of each other
        targets = list(ex.map(lambda w: _resolve(w.get("uri", "")), candidates))
    for web, target in zip(candidates, targets, strict=True):
        if not target:
            dropped.append(web.get("domain") or "unresolved")
        elif (m := ARXIV_ID.search(target)) and _allowed(target, {"arxiv.org"}):
            if m[1] not in arxiv_ids:
                arxiv_ids.append(m[1])
        elif _allowed(target, LINK_ONLY_DOMAINS):
            if target not in links:
                links.append(target)
        else:
            dropped.append(_host(target) or "invalid")

    papers = []
    if arxiv_ids:                                   # the paper's details come from arXiv itself
        try:
            papers = list(arxiv.Client().results(arxiv.Search(id_list=arxiv_ids[:WEB_MAX_SOURCES])))
        except Exception as e:
            log.warning("arXiv verification failed: %s", e)
    links = links[: max(0, WEB_MAX_SOURCES - len(papers))]

    parts = [f"- {p.title} ({', '.join(a.name for a in p.authors[:4])}, {p.published:%Y-%m-%d}) {p.entry_id}\n"
             f"  Abstract: {p.summary[:800]}" for p in papers]
    parts += [f"- link only (content not verified): {u}" for u in links]
    stats = {"queries": meta.get("web_search_queries") or [], "sources": len(chunks), "arxiv_verified": len(papers),
             "links": len(links), "dropped": dropped}
    log.info("web search: %d arXiv verified, %d links, %d dropped", len(papers), len(links), len(dropped),
             extra={"json_fields": {"event": "web_search", "topic": topic[:200], **stats}})
    text = "\n".join(parts) if parts else "(no results: no verified sources found)"
    return text, [p.entry_id for p in papers] + links, stats
