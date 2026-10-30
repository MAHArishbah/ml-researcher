"""Provider-agnostic chat models.

The rest of the code only uses the LangChain chat-model interface (invoke, bind_tools, with_structured_output,
standard usage_metadata), the way an agent talks to a small LLMClient protocol. Which provider sits behind it
is decided here, from "provider:model" strings in .env (MAIN_MODEL, RESEARCHER_MODEL, JUDGE_MODEL, VISION_MODEL).

Supported providers: google_genai (Gemini on Vertex AI) and anthropic (Claude, needs ANTHROPIC_API_KEY).
Provider-specific extras live behind their own small interfaces: sandbox.py (hosted Python) and cache.py
(prompt caching).
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import LOCATION, MAIN_MODEL, PROJECT, RESEARCHER_MODEL
from .ratelimit import PACER

PROVIDERS = ("google_genai", "anthropic")


def _provider_kwargs(provider: str, effort: str | None) -> dict:
    """Map one 'effort' knob (low|medium|high) onto each provider's own setting."""
    if provider == "google_genai":
        kw = {"vertexai": True, "project": PROJECT, "location": LOCATION}
        return {**kw, "thinking_level": effort} if effort else kw
    if provider == "anthropic":
        kw = {"max_tokens": 16000}                         # briefs are long; the SDK default is small
        return {**kw, "output_config": {"effort": effort}} if effort else kw
    raise ValueError(f"unsupported provider {provider!r}; use one of {PROVIDERS}")


def chat_model(spec: str, effort: str | None = None) -> BaseChatModel:
    """spec = "provider:model". All models share one rate limiter and retry policy."""
    provider, name = spec.split(":", 1)
    return init_chat_model(name, model_provider=provider, max_retries=3, rate_limiter=PACER,
                           **_provider_kwargs(provider, effort))


def provider_of(llm) -> str:
    """Provider behind a chat model (also through .bind_tools / .with_structured_output wrappers)."""
    llm = getattr(llm, "bound", llm)
    name = type(llm).__name__
    return {"ChatGoogleGenerativeAI": "google_genai", "ChatAnthropic": "anthropic"}.get(name, name)


def structured(llm, schema):
    """Structured output in native JSON-schema mode: works on Gemini and on Claude with thinking enabled
    (Claude's default, a forced tool call, is not allowed together with thinking)."""
    return llm.with_structured_output(schema, method="json_schema")


MAIN_LLM = chat_model(MAIN_MODEL)
FAST_LLM = chat_model(MAIN_MODEL, effort="low")
RESEARCH_LLM = chat_model(RESEARCHER_MODEL, effort="low")    # tool-calling researchers


def text_of(msg) -> str:
    """Plain text of a LangChain message (content may be a list of blocks, incl. thinking blocks)."""
    t = getattr(msg, "text", None)            # a str (subclass) property in langchain-core 1.x; method before
    if not isinstance(t, str):
        t = t() if callable(t) else None
    return str(t) if t else str(msg.content)
