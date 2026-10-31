"""Index of the user's own code: a second Chroma collection (ml_code_<model>_<dim>) beside the notes.

    python -m ml_researcher.ingest --code                 # CODE_DIRS from .env (default /mnt/e/AI_ML/Prac)

Chunking follows code structure, so a search hit is a whole unit with its location:
  .py      module header (imports, docstring, top-level code), each function, each class (or each method
           of a large class), via the ast module; very long bodies split by lines
  .ipynb   consecutive cells packed together; code + markdown; outputs trimmed, images dropped
  others   (.js .ts .md .yaml .sh ...) blocks between blank lines, packed
Every chunk starts with "# <relative path> :: <symbol> (line N)".

Secrets never leave the machine: secret-looking files are skipped, and before a chunk is embedded or stored,
anything that looks like an API key, token, private key or password assignment is replaced by <redacted>.
"""
import ast
import json
import logging
import os
import re
from pathlib import Path

from .config import CODE_DIRS
from .store_chroma import ChromaIndex

log = logging.getLogger("ml_researcher.codeindex")
CODE_EXTS = {".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs",
             ".sh", ".sql", ".r", ".scala", ".kt", ".yaml", ".yml", ".toml", ".md"}
SKIP_DIRS = {"venv", "env", ".venv", "site-packages", "node_modules", "__pycache__", ".git", ".ipynb_checkpoints",
             "build", "dist", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox", "wandb", "mlruns", ".idea",
             ".vscode"}
MAX_BYTES = {".ipynb": 20_000_000}          # notebooks are mostly outputs, which are trimmed
DEFAULT_MAX_BYTES = 2_000_000
CODE_CHUNK = 3000                           # characters per chunk (a function rarely needs more)
OUTPUT_CHARS = 300                          # notebook output kept per cell
LANG = {".py": "python", ".ipynb": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".sh": "shell", ".md": "markdown", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml"}

# ---------- secrets ----------
_SECRET_FILE = re.compile(r"^\.env|\.(pem|key|p12|pfx|jks)$|^id_(rsa|ed25519|ecdsa)|^\.netrc$|"
                          r"(credential|secret|service[_-]?account).*\.(json|ya?ml|txt)$", re.IGNORECASE)
_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                              # Google API key
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{30,}"),
    re.compile(r"\bsk-(?:ant-|proj-)?[0-9A-Za-z_\-]{20,}"),                # OpenAI / Anthropic
    re.compile(r"\bhf_[0-9A-Za-z]{30,}"),                                 # Hugging Face
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                  # AWS access key id
    re.compile(r"\bxox[abposr]-[0-9A-Za-z\-]{10,}"),                      # Slack
]
_ASSIGNED = re.compile(r"(?i)\b([\w-]*(?:api[_-]?key|secret|token|passw(?:or)?d|pwd)[\w-]*)(\s*[:=]\s*)"
                       r"(['\"])([^'\"\n]{6,})\3")


_SPECIAL_TOKEN = re.compile(r"(?i)(pad|eos|bos|unk|sep|cls|mask|start|end|special)_?tokens?$|token_?(id|ids|type)")
_SECRET_VALUE = re.compile(r"(?=[^\s<>{}]*\d)(?=[^\s<>{}]*[A-Za-z])[^\s<>{}]{12,}")   # 12+ chars, letters+digits


def is_secret_file(p: Path) -> bool:
    return bool(_SECRET_FILE.search(p.name))


def redact(text: str) -> tuple[str, int]:
    """Replace secret-looking values with <redacted>. Returns (text, how many were replaced)."""
    n = 0
    for pat in _SECRETS:
        text, k = pat.subn("<redacted>", text)
        n += k
    hits = 0

    def assigned(m):          # name says secret AND value looks like one (not pad_token = "<eos>")
        nonlocal hits
        if _SPECIAL_TOKEN.search(m[1]) or not _SECRET_VALUE.fullmatch(m[4]):
            return m[0]
        hits += 1
        return f"{m[1]}{m[2]}{m[3]}<redacted>{m[3]}"

    text = _ASSIGNED.sub(assigned, text)
    return text, n + hits


# ---------- walking ----------
def walk(root: Path):
    """Every code file under root, skipping environments, caches, hidden folders and secret files."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and "site-packages" not in d]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() not in CODE_EXTS or is_secret_file(p) or f.startswith("~$"):
                continue
            try:
                if p.stat().st_size <= MAX_BYTES.get(p.suffix.lower(), DEFAULT_MAX_BYTES):
                    yield p
            except OSError:
                pass


def _rel(p: Path) -> str:
    for root in CODE_DIRS:
        try:
            return str(p.relative_to(root))
        except ValueError:
            continue
    return p.name


# ---------- chunking ----------
def _split_lines(text: str, start: int) -> list:
    """(text, first line) pieces of at most CODE_CHUNK characters, cut at line boundaries."""
    pieces, buf, first = [], [], start
    for i, line in enumerate(text.splitlines(), start):
        if buf and sum(len(x) + 1 for x in buf) + len(line) > CODE_CHUNK:
            pieces.append(("\n".join(buf), first))
            buf, first = [], i
        buf.append(line[:CODE_CHUNK])
    return pieces + ([("\n".join(buf), first)] if buf else [])


def _python_units(src: str) -> list:
    """[(symbol, first line, code)] for a Python file: header, functions, classes or their methods."""
    tree = ast.parse(src)
    lines = src.splitlines()
    seg = lambda node: "\n".join(lines[node.lineno - 1 - len(node.decorator_list) if hasattr(node, "decorator_list")  # noqa: E731
                                       else node.lineno - 1: node.end_lineno])
    units, header = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append((node.name, node.lineno, seg(node)))
        elif isinstance(node, ast.ClassDef):
            code = seg(node)
            if len(code) <= CODE_CHUNK:
                units.append((node.name, node.lineno, code))
            else:                                    # big class: its header, then each method on its own
                body_start = node.body[0].lineno - 1 if node.body else node.end_lineno
                units.append((node.name, node.lineno, "\n".join(lines[node.lineno - 1:body_start])))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        units.append((f"{node.name}.{sub.name}", sub.lineno, seg(sub)))
        else:
            header.append(seg(node))
    if header:
        units.insert(0, ("<module>", 1, "\n".join(header)))
    return units


def _notebook_units(raw: str) -> list:
    """[(symbol, first cell number, text)] packing consecutive cells, outputs trimmed."""
    cells = json.loads(raw).get("cells", [])
    units, buf, first = [], [], 1
    for n, cell in enumerate(cells, 1):
        src = "".join(cell.get("source", []))
        out = []
        for o in cell.get("outputs", []) if cell.get("cell_type") == "code" else []:
            text = o.get("text") or (o.get("data") or {}).get("text/plain") or ""
            out.append("".join(text) if isinstance(text, list) else str(text))
        block = src.strip()
        if out and "".join(out).strip():
            block += "\n# output:\n" + "".join(out).strip()[:OUTPUT_CHARS]
        if not block:
            continue
        if buf and sum(len(b) for b in buf) + len(block) > CODE_CHUNK:
            units.append((f"cells {first}-{n - 1}", first, "\n\n".join(buf)))
            buf, first = [], n
        buf.append(block[:CODE_CHUNK])
    if buf:
        units.append((f"cells {first}-{len(cells)}", first, "\n\n".join(buf)))
    return units


def _block_units(src: str) -> list:
    """[(symbol, first line, text)] for any other file: blank-line blocks packed up to CODE_CHUNK."""
    units, buf, first, line = [], [], 1, 1
    for block in re.split(r"\n\s*\n", src):
        if buf and sum(len(b) + 2 for b in buf) + len(block) > CODE_CHUNK:
            units.append((f"lines {first}-{line - 1}", first, "\n\n".join(buf)))
            buf, first = [], line
        buf.append(block)
        line += block.count("\n") + 2
    if buf:
        units.append((f"lines {first}-{line}", first, "\n\n".join(buf)))
    return units


def pieces(p: Path) -> list:
    """[(chunk text, metadata)] for one file, secrets redacted."""
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    ext = p.suffix.lower()
    try:
        units = (_python_units(src) if ext == ".py" else _notebook_units(src) if ext == ".ipynb"
                 else _block_units(src))
    except (SyntaxError, ValueError):                        # broken .py / notebook: plain blocks instead
        units = _block_units(src)
    rel, out, redacted = _rel(p), [], 0
    for symbol, line, code in units:
        for k, (part, first) in enumerate(_split_lines(code, line) if len(code) > CODE_CHUNK else [(code, line)]):
            if not part.strip():
                continue
            part, n = redact(part)
            redacted += n
            name = symbol if k == 0 else f"{symbol} (part {k + 1})"
            out.append((f"# {rel} :: {name} (line {first})\n{part}",
                        {"symbol": name, "line": first, "lang": LANG.get(ext, ext.lstrip(".")), "rel_path": rel}))
    if redacted:
        log.warning("redacted %d secret-looking value(s) in %s", redacted, rel, extra={"json_fields": {
            "event": "code_redacted", "path": rel, "count": redacted}})
    return out


CODE = ChromaIndex("code", pieces=pieces,
                   title=lambda p, text, extra: f"title: {extra['rel_path']} :: {extra['symbol']} | text: {text}")


# module API used by ingest.py and tools.py (same names as the notes store)
def count() -> int:
    return CODE.count()


def indexed_files() -> dict:
    return CODE.indexed_files()


def delete_paths(paths) -> None:
    CODE.delete_paths(paths)


def upsert_file(p: Path, mtime: float) -> int:
    return CODE.upsert_file(p, mtime)


def search(query: str, top_k: int = 5, file_type: str = "") -> list:
    return CODE.search(query, top_k, file_type)
