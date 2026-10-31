"""URL opener for /read questions: fetch ONE page the user typed, as plain text, behind strict guardrails.

    /read https://example.com/post  what does it say about superposition?

Guardrails:
  - only the URL the user typed (the model can never ask for a fetch, links on the page are never followed);
  - https only; the host (and every redirect hop, at most MAX_REDIRECTS) must resolve to public internet
    addresses: localhost, private, link-local and reserved ranges are refused (no access to this machine
    or the local network). The connection is made to the exact address that was checked (pinned), so a
    DNS answer that changes between check and connect (DNS rebinding) can't redirect it; TLS still
    verifies the certificate for the real hostname;
  - web pages (up to 10 MB) and PDFs (up to 15 MB) only; per-read timeouts AND a total deadline, so a
    server that drips bytes can't hold the turn open;
  - only text is kept (scripts, styles, navigation, forms dropped), capped at MAX_CHARS (then condensed to ~5k tokens in graph.py); the caller treats it
    as untrusted data (wrapped, scanned for injection phrases).
"""
import ipaddress
import logging
import re
import socket
import time
from urllib.parse import urljoin, urlparse

import certifi
import lxml.html
import pymupdf
import urllib3

log = logging.getLogger("ml_researcher.urlreader")
READ_CMD = re.compile(r"(?<!\S)/read(?!\S)", re.IGNORECASE)   # the /read command, anywhere in the message
URL = re.compile(r"https?://[^\s<>\"']+")
MAX_BYTES = {"html": 10_000_000, "pdf": 15_000_000}    # pages embed images/scripts; only text is kept
MAX_CHARS = 200_000                      # read long articles whole; graph.py condenses them to ~5k tokens
MAX_REDIRECTS = 3
TIMEOUT = urllib3.Timeout(connect=10, read=20)   # per connect / per read (s)
TOTAL_SECONDS = 60                       # whole download, however slowly bytes arrive
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


def _check_host(url: str) -> str:
    """https + every address the host resolves to must be public. Returns the address to connect to."""
    parts = urlparse(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ReadError("only https:// links are allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ReadError(f"cannot resolve {parts.hostname}") from None
    ips = [ipaddress.ip_address(info[4][0]) for info in infos]
    for ip in ips:
        if not ip.is_global or ip.is_multicast:
            raise ReadError(f"{parts.hostname} points to a non-public address ({ip}); refused")
    return str(sorted(ips, key=lambda ip: ip.version)[0])          # prefer IPv4


def _get(url: str, ip: str):
    """GET pinned to the checked address: TCP goes to `ip`, while SNI, the Host header and certificate
    verification use the URL's hostname. No new DNS lookup happens, so DNS rebinding can't swap the target."""
    parts = urlparse(url)
    pool = urllib3.HTTPSConnectionPool(ip, port=parts.port or 443, server_hostname=parts.hostname,
                                       assert_hostname=parts.hostname, cert_reqs="CERT_REQUIRED",
                                       ca_certs=certifi.where(), timeout=TIMEOUT, retries=False)
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return pool.urlopen("GET", path, headers={**HEADERS, "Host": parts.netloc}, redirect=False,
                        preload_content=False)


def _download(url: str) -> tuple[str, bytes, str]:
    """(final url, body, content type) following at most MAX_REDIRECTS hops, each one re-checked and pinned."""
    deadline = time.monotonic() + TOTAL_SECONDS
    for _ in range(MAX_REDIRECTS + 1):
        try:
            r = _get(url, _check_host(url))
        except urllib3.exceptions.HTTPError as e:
            raise ReadError(f"connection failed: {type(e).__name__}") from None
        try:
            if r.status in (301, 302, 303, 307, 308):
                url = urljoin(url, r.headers.get("Location", ""))
                continue
            if r.status != 200:
                raise ReadError(f"HTTP {r.status}")
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype not in HTML_TYPES and ctype != "application/pdf":
                raise ReadError(f"unsupported content type {ctype or 'unknown'}")
            limit = MAX_BYTES["pdf" if ctype == "application/pdf" else "html"]
            if int(r.headers.get("Content-Length") or 0) > limit:           # refuse before downloading
                raise ReadError(f"larger than {limit // 1_000_000} MB")
            body = bytearray()
            try:
                for part in r.stream(64_000):
                    body += part
                    if len(body) > limit:
                        raise ReadError(f"larger than {limit // 1_000_000} MB")
                    if time.monotonic() > deadline:
                        raise ReadError(f"download took longer than {TOTAL_SECONDS} s")
            except urllib3.exceptions.HTTPError as e:
                raise ReadError(f"download failed: {type(e).__name__}") from None
            return url, bytes(body), ctype
        finally:
            r.release_conn()
    raise ReadError(f"more than {MAX_REDIRECTS} redirects")


def _html_text(body: bytes) -> tuple[str, str]:
    doc = lxml.html.fromstring(body)
    for el in doc.xpath(DROP):
        el.drop_tree()
    title = (doc.findtext(".//title") or "").strip()
    # outermost blocks only: a <p> inside an <li> or <td> is already part of that block's text
    blocks = [re.sub(r"\s+", " ", el.text_content()).strip() for el in doc.iter(*BLOCKS)
              if not any(a.tag in BLOCKS for a in el.iterancestors())]
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
