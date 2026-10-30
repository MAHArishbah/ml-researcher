from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent.parent / ".env")


PROJECT=os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION=os.getenv("GOOGLE_CLOUD_LOCATION","global")

def _spec(value: str) -> str:
    """Models are "provider:model" (e.g. "anthropic:claude-sonnet-5-5"); a bare name means Gemini."""
    return value if ":" in value else f"google_genai:{value}"


MAIN_MODEL = _spec(os.getenv("MAIN_MODEL") or os.getenv("RESEARCH_MODEL", "gemini-3.8-flash"))   # router, planner,
RESEARCHER_MODEL = _spec(os.getenv("RESEARCHER_MODEL") or MAIN_MODEL)     # reviewer, synthesizer, answers | researchers
JUDGE_MODEL = _spec(os.getenv("JUDGE_MODEL") or MAIN_MODEL)               # evaluate: golden-set writer + DeepEval judge
VISION_MODEL = _spec(os.getenv("VISION_MODEL") or RESEARCHER_MODEL)       # image captions during ingest
WEB_SEARCH = os.getenv("WEB_SEARCH", "true").lower() == "true"          # allow /web questions (Gemini grounding)
CODE_SANDBOX = os.getenv("CODE_SANDBOX", "auto").lower()                  # auto (MAIN_MODEL's provider) | google_genai
                                                                          # | anthropic | none

VECTOR_BACKEND=os.getenv("VECTOR_BACKEND","chroma").lower()
DOCS_DIR = Path(os.getenv("LOCAL_DOCS_DIR", "/mnt/e/AI_ML"))
CHROMA_DIR = Path(os.path.expanduser(os.getenv("CHROMA_DIR", "~/.ml_researcher/chroma")))
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
FILE_WORKERS = int(os.getenv("FILE_WORKERS", "2"))
HYBRID = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
LLM_RPS = float(os.getenv("LLM_RPS") or os.getenv("GEMINI_RPS", "5"))  # max model calls/s per process; 0 = no limit
# Max silence (s) on a model connection before the call is abandoned and retried (same model, same prompt).
# Streaming answers that keep sending tokens never hit it; only a request that gets no reply does.
SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", "30000"))  # follow-ups: summarise older turns
                                                                          # once the conversation is this long
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))            # answers, briefs, captions, judge (thinking can take ~50 s)
FAST_LLM_TIMEOUT = float(os.getenv("FAST_LLM_TIMEOUT", "45"))  # router, planner, reviewer, researchers (normally 2-10 s)
SANDBOX_TIMEOUT = float(os.getenv("SANDBOX_TIMEOUT", "180"))   # Python checks: code runs before any reply comes back
BATCH_RPS = float(os.getenv("BATCH_RPS", "1"))       # stricter pace for batch jobs (ingest, evaluate)
RERANK = os.getenv("RERANK", "true").lower() == "true"            # needs COHERE_API_KEY
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v5.0-pro")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))     # fused hits sent to the reranker
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CHECKPOINT_DB = Path(os.path.expanduser(os.getenv("CHECKPOINT_DB", "~/.ml_researcher/checkpoints.sqlite")))

LOG_DIR = Path(os.path.expanduser(os.getenv("LOG_DIR", "~/.ml_researcher/logs")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_GCP = os.getenv("LOG_TO_GCP", "false").lower() == "true"   # also ship logs to Cloud Logging
LOG_NAME = os.getenv("LOG_NAME", "ml-researcher")                  # Cloud Logging log name
LOG_PROMPTS = os.getenv("LOG_PROMPTS", "true").lower() == "true"    # include prompt text in request logs


RAG_LOCATION = os.getenv("RAG_LOCATION", "europe-west4")
RAG_CORPUS_NAME = os.getenv("RAG_CORPUS_NAME", "ml-notes")


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")