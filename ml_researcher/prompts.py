"""Loads every model prompt from prompts.md, so changing a prompt is a one-file change.

    prompt("SYNTH")                 -> the text, exactly as written
    prompt("ROUTER", topic="...")   -> the text with its {placeholders} filled in
"""
import re
from pathlib import Path

PROMPTS_FILE = Path(__file__).with_name("prompts.md")
_MARKER = re.compile(r"^=== ([A-Z0-9_]+) ===[ \t]*$", re.MULTILINE)


def _load(path: Path = PROMPTS_FILE) -> dict:
    parts = _MARKER.split(path.read_text(encoding="utf-8"))     # [preamble, name, body, name, body, ...]
    prompts = {}
    for name, body in zip(parts[1::2], parts[2::2], strict=True):
        lines = body.strip("\n").splitlines()
        while lines and lines[0].startswith("> "):              # notes for the reader, not sent to the model
            lines.pop(0)
        if name in prompts:
            raise ValueError(f"prompt {name} is defined twice in {path.name}")
        prompts[name] = "\n".join(lines).strip()
    return prompts


PROMPTS = _load()


def prompt(name: str, **values) -> str:
    """The named prompt; with values, its {placeholders} filled in (a missing value raises KeyError)."""
    text = PROMPTS[name]
    return text.format(**values) if values else text
