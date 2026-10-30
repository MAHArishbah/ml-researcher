"""URL opener for /read questions: fetch ONE page the user typed, as plain text, behind strict guardrails.

    /read https://example.com/post  what does it say about superposition?

Guardrails:
  - only the URL the user typed (the model can never ask for a fetch, links on the page are never followed);
  - https only; the host (and every redirect hop, at most MAX_REDIRECTS) must resolve to public internet
    addresses: localhost, private, link-local and reserved ranges are refused (no access to this machine
    or the local network);
  - web pages (up to 10 MB) and PDFs (up to 15 MB) only; connect/read timeouts;
  - only text is kept (scripts, styles, navigation, forms dropped), capped at MAX_CHARS (then condensed to ~5k tokens in graph.py); the caller treats it
    as untrusted data (wrapped, scanned for injection phrases).
"""
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import lxml.html
import pymupdf
import requests

log = logging.getLogger("ml_researcher.urlreader")
READ_CMD = re.compile(r"(?<!\S)/read(?!\S)", re.IGNORECASE)   # the /read command, anywhere in the message
URL = re.compile(r"https?://[^\s<>\"']+")
MAX_BYTES = {"html": 10_000_000, "pdf": 15_000_000}    # pages embed images/scripts; only text is kept
MAX_CHARS = 200_000                      # read long articles whole; graph.py condenses them to ~5k tokens
MAX_REDIRECTS = 3
TIMEOUT = (10, 20)                       # connect, read (s)
HEADERS = {"User-Agent": "ml-researcher/1.0 (personal study assistant; reads one page on request)"}
HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
DROP = "//script|//style|//noscript|//svg|//nav|//footer|//form|//iframe|//button|//aside"
BLOCKS = ("h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote", "figcaption", "td")


class ReadError(Exception):
    """The page can't be read safely; the message says why (shown to the model and logged)."""


def parse_read(message: str) -> tuple[str, str] | None:
    """A message containing the /read command and a link, in any order -> (url, question), else None.
    Still opt-in: a link without /read is never fetched. With several links, the first one is read."""
    message = message or ""
    url = URL.search(message)
    if not READ_CMD.search(message) or not url:
        return None
    question = URL.sub(" ", READ_CMD.sub(" ", message), count=1)
    return url[0].rstrip(").,;"), re.sub(r"\s+", " ", question).strip()


def _check_host(url: str) -> None:
    """https + every address the host resolves to must be public."""
    parts = urlparse(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ReadError("only https:// links are allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ReadError(f"cannot resolve {parts.hostname}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ReadError(f"{parts.hostname} points to a non-public address ({ip}); refused")


def _download(url: str) -> tuple[str, bytes, str]:
    """(final url, body, content type) following at most MAX_REDIRECTS hops, each one re-checked."""
    for _ in range(MAX_REDIRECTS + 1):
        _check_host(url)
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=False) as r:
            if r.is_redirect:
                url = urljoin(url, r.headers.get("Location", ""))
                continue
            if r.status_code != 200:
                raise ReadError(f"HTTP {r.status_code}")
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype not in HTML_TYPES and ctype != "application/pdf":
                raise ReadError(f"unsupported content type {ctype or 'unknown'}")
            limit = MAX_BYTES["pdf" if ctype == "application/pdf" else "html"]
            body = b""
            for part in r.iter_content(64_000):
                body += part
                if len(body) > limit:
                    raise ReadError(f"larger than {limit // 1_000_000} MB")
            return url, body, ctype
    raise ReadError(f"more than {MAX_REDIRECTS} redirects")


def _html_text(body: bytes) -> tuple[str, str]:
    doc = lxml.html.fromstring(body)
    for el in doc.xpath(DROP):
        el.drop_tree()
    title = (doc.findtext(".//title") or "").strip()
    blocks = [re.sub(r"\s+", " ", el.text_content()).strip() for el in doc.iter(*BLOCKS)]
    text = "\n\n".join(b for b in blocks if len(b) > 1)
    if len(text) < 200:                          # unusual markup: fall back to all visible text
        text = re.sub(r"\n\s*\n+", "\n\n", doc.text_content())
    return title, text


def _pdf_text(body: bytes) -> tuple[str, str]:
    with pymupdf.open(stream=body, filetype="pdf") as doc:
        return (doc.metadata or {}).get("title", "") or "", "\n\n".join(p.get_text().strip() for p in doc)


def read(url: str) -> dict:
    """{'url', 'title', 'text', 'chars', 'truncated'}; raises ReadError when the page can't be read safely."""
    final, body, ctype = _download(url)
    title, text = _pdf_text(body) if ctype == "application/pdf" else _html_text(body)
    text = text.strip()
    if not text:
        raise ReadError("no readable text on the page")
    page = {"url": final, "title": title[:200], "text": text[:MAX_CHARS], "chars": len(text),
            "truncated": len(text) > MAX_CHARS}
    log.info("read %s (%d chars)", final, len(text), extra={"json_fields": {
        "event": "url_read", "url": url, "final_url": final, "bytes": len(body), "chars": len(text),
        "content_type": ctype}})
    return page
