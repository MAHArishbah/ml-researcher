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
CODE_DIRS = [Path(os.path.expanduser(p.strip())) for p in os.getenv("CODE_DIRS", "/mnt/e/AI_ML/Prac").split(",")
             if p.strip()]                                            # your code, indexed with ingest --code
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
REPLY_TIMEOUT = float(os.getenv("REPLY_TIMEOUT", "90"))          # answer + synthesizer
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))            # captions, judge, article condensing, summaries
FAST_LLM_TIMEOUT = float(os.getenv("FAST_LLM_TIMEOUT", "45"))  # router, planner, reviewer, researchers (normally 2-10 s)
SANDBOX_TIMEOUT = float(os.getenv("SANDBOX_TIMEOUT", "180"))   # Python checks: code runs before any reply comes back
BATCH_RPS = float(os.getenv("BATCH_RPS", "1"))       # stricter pace for batch jobs (ingest, evaluate)
RERANK = os.getenv("RERANK", "true").lower() == "true"            # needs COHERE_API_KEY
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v5.0-pro")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))     # fused hits sent to the reranker
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CHECKPOINT_DB = Path(os.path.expanduser(os.getenv("CHECKPOINT_DB", "~/.ml_researcher/checkpoints.sqlite")))

MEMORY_FILE = Path(os.path.expanduser(os.getenv("MEMORY_FILE", "~/.ml_researcher/memory.json")))  # profile facts
MEMORY_AUTO = os.getenv("MEMORY_AUTO", "true").lower() == "true"     # mine every finished turn for memories
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "8"))                    # memories recalled per /memory turn
MEMORY_CANDIDATES = int(os.getenv("MEMORY_CANDIDATES", "200"))        # nearest memories scored per recall
MEMORY_DECAY = float(os.getenv("MEMORY_DECAY", "0.995"))              # recency = DECAY ** hours since last use
MEMORY_WEIGHTS = tuple(float(w) for w in os.getenv("MEMORY_WEIGHTS", "1,1,1").split(","))  # recency, importance, relevance
MEMORY_REFLECT_AT = int(os.getenv("MEMORY_REFLECT_AT", "150"))        # importance sum that triggers a reflection

LOG_DIR = Path(os.path.expanduser(os.getenv("LOG_DIR", "~/.ml_researcher/logs")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_GCP = os.getenv("LOG_TO_GCP", "false").lower() == "true"   # also ship logs to Cloud Logging
LOG_NAME = os.getenv("LOG_NAME", "ml-researcher")                  # Cloud Logging log name
LOG_PROMPTS = os.getenv("LOG_PROMPTS", "true").lower() == "true"    # include prompt text in request logs
DASHBOARD_TZ = os.getenv("DASHBOARD_TZ", "UTC")                    # times on the dashboard page, e.g. Asia/Kolkata


RAG_LOCATION = os.getenv("RAG_LOCATION", "europe-west4")
RAG_CORPUS_NAME = os.getenv("RAG_CORPUS_NAME", "ml-notes")


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

DATA_DIR = Path(os.path.expanduser("~/.ml_researcher"))


def private_dir(path: Path) -> Path:
    """Create `path` (and parents). Folders in the app's own DATA_DIR are made owner-only: conversations and logs
    hold your questions, and files inside a 700 folder are protected whatever their own mode. A folder set
    elsewhere in .env is created but its permissions are left alone."""
    path.mkdir(parents=True, exist_ok=True)
    if path == DATA_DIR or DATA_DIR in path.parents:
        try:
            path.chmod(0o700)
        except OSError:                              # e.g. a folder on a Windows drive: leave it
            pass
    return path
