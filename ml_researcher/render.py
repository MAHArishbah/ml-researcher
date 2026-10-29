"""Make LLM markdown render correctly in Streamlit (KaTeX math, literal $ and ~)."""
import re

_CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)     # never touch code
_MATH = re.compile(
    r"\$\$(?P<d1>.+?)\$\$"                              # $$ display $$
    r"|\\\[(?P<d2>.+?)\\\]"                             # \[ display \]
    r"|(?P<env>\\begin\{(?P<name>equation|align|gather|multline)\*?\}.*?\\end\{(?P=name)\*?\})"  # bare environment
    r"|\\\((?P<i1>[^\n]+?)\\\)"                         # \( inline \)
    # $inline$ (pandoc rules): opening $ followed by non-space, closing $ preceded by
    # non-space and not followed by a digit -> "$5 and $10" stays text
    r"|(?<![\\$\w])\$(?=[^\s$])(?P<i2>(?:\\.|[^$\\\n])+?)(?<=[^\s\\])\$(?!\d)",
    re.S)


def _display(body: str) -> str:
    return f"\n\n$$\n{body.strip()}\n$$\n\n"


def _inline(body: str) -> str:
    body = body.strip().replace(r"\|", r"\Vert ").replace("|", r"\vert ")  # "|" would split table cells
    return f"${body}$"


def _escape(text: str) -> str:
    return re.sub(r"(?<!\\)([$~])", r"\\\1", text)     # stray $ (currency) and ~ (strikethrough)


def _fix_prose(text: str) -> str:
    out, last = [], 0
    for m in _MATH.finditer(text):
        out.append(_escape(text[last:m.start()]))
        if m["d1"] is not None or m["d2"] is not None:
            out.append(_display(m["d1"] if m["d1"] is not None else m["d2"]))
        elif m["env"] is not None:
            out.append(_display(m["env"]))
        else:
            out.append(_inline(m["i1"] if m["i1"] is not None else m["i2"]))
        last = m.end()
    out.append(_escape(text[last:]))
    s = "".join(out)
    return re.sub(r"[ \t]+(\n\n\$\$\n)|(\n\$\$\n\n)[ \t]+", lambda m: m[1] or m[2], s)  # spaces around blocks


def to_streamlit_md(md: str) -> str:
    """Normalise \\( \\) / \\[ \\] / bare environments to $ / $$, keep currency and ~ literal."""
    parts = _CODE.split(md)
    fixed = "".join(p if i % 2 else _fix_prose(p) for i, p in enumerate(parts))
    return re.sub(r"\n{3,}", "\n\n", fixed).strip()
