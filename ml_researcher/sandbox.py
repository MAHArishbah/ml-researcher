"""Hosted Python sandboxes behind one small interface.

Model-written code only ever runs in the provider's own sandbox (no network, no access to this machine):
  google_genai -> Gemini code execution        anthropic -> Claude code execution tool
Nothing is executed locally. With CODE_SANDBOX=none (or an unsupported provider) Python checks are disabled
and the reviewer's python follow-ups are dropped.
"""
import warnings
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from .config import CODE_SANDBOX, MAIN_MODEL, SANDBOX_TIMEOUT

# langchain-google-genai warns on every reply that contains executed code that the result block may be
# absent; GeminiSandbox.run already handles a missing result, so the notice is just terminal noise
warnings.filterwarnings("ignore", message=r"\s*Warning: Output may vary each run", category=UserWarning)


@dataclass
class SandboxRun:
    code: str
    output: str
    prose: str


class CodeSandbox(Protocol):
    name: str

    def run(self, system: str, prompt: str) -> SandboxRun: ...


def _blocks(content) -> list:
    return content if isinstance(content, list) else [content]


class GeminiSandbox:
    """Gemini's built-in code execution: code and results come back as executable_code /
    code_execution_result blocks."""

    def __init__(self, spec: str):
        from .llm import chat_model
        self.name = f"gemini code execution ({spec})"
        self.llm = chat_model(spec, effort="low", timeout=SANDBOX_TIMEOUT).bind_tools([{"code_execution": {}}])

    def run(self, system: str, prompt: str) -> SandboxRun:
        reply = self.llm.invoke([SystemMessage(system), HumanMessage(prompt)])
        code, out, prose = [], [], []
        for b in _blocks(reply.content):
            if isinstance(b, str):
                prose.append(b)
            elif b.get("type") == "executable_code":
                code.append(b.get("executable_code", ""))
            elif b.get("type") == "code_execution_result":
                out.append(b.get("code_execution_result", ""))
            elif b.get("type") == "text":
                prose.append(b.get("text", ""))
        return SandboxRun("\n\n".join(code), "\n".join(out), "".join(prose))


class AnthropicSandbox:
    """Claude's server-side code execution tool: server_tool_use blocks carry the code (or shell command),
    *code_execution_tool_result blocks carry stdout/stderr."""

    TOOL = {"type": "code_execution_20260521", "name": "code_execution"}

    def __init__(self, spec: str):
        from .llm import chat_model
        self.name = f"claude code execution ({spec})"
        self.llm = chat_model(spec, effort="low", timeout=SANDBOX_TIMEOUT).bind_tools([self.TOOL])

    def run(self, system: str, prompt: str) -> SandboxRun:
        reply = self.llm.invoke([SystemMessage(system), HumanMessage(prompt)])
        code, out, prose = [], [], []
        for b in _blocks(reply.content):
            if isinstance(b, str):
                prose.append(b)
                continue
            kind = b.get("type", "")
            if kind == "server_tool_use":
                inp = b.get("input") or {}
                code.append(inp.get("code") or inp.get("command") or inp.get("file_text") or "")
            elif kind.endswith("code_execution_tool_result"):
                res = b.get("content") or {}
                for r in res if isinstance(res, list) else [res]:
                    if isinstance(r, dict):
                        out.append("\n".join(x for x in (r.get("stdout"), r.get("stderr")) if x))
            elif kind == "text":
                prose.append(b.get("text", ""))
        return SandboxRun("\n\n".join(c for c in code if c), "\n".join(o for o in out if o), "".join(prose))


ADAPTERS = {"google_genai": GeminiSandbox, "anthropic": AnthropicSandbox}


def get_sandbox() -> CodeSandbox | None:
    """CODE_SANDBOX=auto uses MAIN_MODEL's provider; none (or no adapter) disables Python checks."""
    provider = MAIN_MODEL.split(":", 1)[0] if CODE_SANDBOX == "auto" else CODE_SANDBOX
    if provider == "none" or provider not in ADAPTERS:
        return None
    model = MAIN_MODEL if MAIN_MODEL.startswith(provider + ":") else None
    if model is None:                       # explicit sandbox provider different from MAIN_MODEL's
        raise ValueError(f"CODE_SANDBOX={provider} needs MAIN_MODEL from the same provider (got {MAIN_MODEL})")
    return ADAPTERS[provider](model)
