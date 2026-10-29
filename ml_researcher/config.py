from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent.parent / ".env")


PROJECT=os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION=os.getenv("GOOGLE_CLOUD_LOCATION","global")
MODEL=os.getenv("RESEARCH_MODEL","gemini-3.8-flash")
RESEARCHER_MODEL=os.getenv("RESEARCHER_MODEL",MODEL)

VECTOR_BACKEND=os.getenv("VECTOR_BACKEND","chroma").lower()
DOCS_DIR = Path(os.getenv("LOCAL_DOCS_DIR", "/mnt/e/AI_ML"))
CHROMA_DIR = Path(os.path.expanduser(os.getenv("CHROMA_DIR", "~/.ml_researcher/chroma")))
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
FILE_WORKERS = int(os.getenv("FILE_WORKERS", "2"))
HYBRID = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
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