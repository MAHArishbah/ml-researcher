"""Provider-agnostic chat models.

The rest of the code only uses the LangChain chat-model interface (invoke, bind_tools, with_structured_output,
standard usage_metadata), the way an agent talks to a small LLMClient protocol. Which provider sits behind it
is decided here, from "provider:model" strings in .env (MAIN_MODEL, RESEARCHER_MODEL, JUDGE_MODEL, VISION_MODEL).

Supported providers: google_genai (Gemini on Vertex AI) and anthropic (Claude, needs ANTHROPIC_API_KEY).
Provider-specific extras live behind their own small interfaces: sandbox.py (hosted Python) and cache.py
(prompt caching).
"""
import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import FAST_LLM_TIMEOUT, LLM_TIMEOUT, LOCATION, MAIN_MODEL, PROJECT, REPLY_TIMEOUT, RESEARCHER_MODEL
from .ratelimit import PACER

PROVIDERS = ("google_genai", "anthropic")
SERVER_DEADLINE = 600              # s: Google-side limit for a whole request (a long answer must never hit it)
# Close pooled connections idle for over 20 s instead of reusing them. WSL's NAT silently drops idle TCP
# connections; a request sent on one is never delivered and just waits out the silence timeout (then the
# retry, on a new connection, answers in seconds). A fresh connection costs ~50 ms.
FRESH_CONNECTIONS = httpx.Limits(keepalive_expiry=20)


def _provider_kwargs(provider: str, effort: str | None) -> dict:
    """Map one 'effort' knob (low|medium|high) onto each provider's own setting."""
    if provider == "google_genai":
        # The SDK also sends `timeout` to Google as X-Server-Timeout, a deadline on the WHOLE request, which would
        # cut long streamed answers. Setting the header ourselves keeps Google's deadline generous while `timeout`
        # still limits silence on the connection locally.
        kw = {"vertexai": True, "project": PROJECT, "location": LOCATION,
              "additional_headers": {"X-Server-Timeout": str(SERVER_DEADLINE)},
              "client_args": {"limits": FRESH_CONNECTIONS}}
        return {**kw, "thinking_level": effort} if effort else kw
    if provider == "anthropic":
        kw = {"max_tokens": 16000}                         # briefs are long; the SDK default is small
        return {**kw, "output_config": {"effort": effort}} if effort else kw
    raise ValueError(f"unsupported provider {provider!r}; use one of {PROVIDERS}")


def chat_model(spec: str, effort: str | None = None, timeout: float = LLM_TIMEOUT) -> BaseChatModel:
    """spec = "provider:model". All models share one rate limiter and retry policy.
    timeout = longest silence on the connection (s); a call that gets no reply for that long is abandoned
    and retried by the provider SDK (3 attempts in all), instead of hanging for minutes."""
    provider, name = spec.split(":", 1)
    return init_chat_model(name, model_provider=provider, max_retries=3, rate_limiter=PACER, timeout=timeout,
                           **_provider_kwargs(provider, effort))


def provider_of(llm) -> str:
    """Provider behind a chat model (also through .bind_tools / .with_structured_output wrappers)."""
    llm = getattr(llm, "bound", llm) #returns langchain's runnablebinding having real model in .bound
    name = type(llm).__name__
    return {"ChatGoogleGenerativeAI": "google_genai", "ChatAnthropic": "anthropic"}.get(name, name)


def structured(llm, schema):
    """Structured output in native JSON-schema mode: works on Gemini and on Claude with thinking enabled
    (Claude's default, a forced tool call, is not allowed together with thinking)."""
    return llm.with_structured_output(schema, method="json_schema")


# writes what the user reads (briefs, answers); it thinks silently before its first token, so its silence
# limit is REPLY_TIMEOUT (90 s)
MAIN_LLM = chat_model(MAIN_MODEL, timeout=REPLY_TIMEOUT)
FAST_LLM = chat_model(MAIN_MODEL, effort="low", timeout=FAST_LLM_TIMEOUT)
RESEARCH_LLM = chat_model(RESEARCHER_MODEL, effort="low", timeout=FAST_LLM_TIMEOUT)    # tool-calling researchers


def text_of(msg) -> str:
    """Plain text of a LangChain message (content may be a list of blocks, incl. thinking blocks)."""
    t = getattr(msg, "text", None)            # a str (subclass) property in langchain-core 1.x; method before
    if not isinstance(t, str):
        t = t() if callable(t) else None
    return str(t) if t else str(msg.content)
