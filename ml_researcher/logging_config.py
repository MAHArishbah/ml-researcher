"""Logging setup: JSON-lines file (+ optional Google Cloud Logging) for the ml_researcher logger.

Log structured fields with:  log.info("msg", extra={"json_fields": {...}})
The same fields land in the local file and in Cloud Logging's jsonPayload.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from .config import LOG_DIR, LOG_LEVEL, LOG_NAME, LOG_TO_GCP, PROJECT

LOGGER = "ml_researcher"
LOG_FILE = LOG_DIR / "agent.jsonl"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, message + json_fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        entry.update(getattr(record, "json_fields", None) or {})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class TurnIds(logging.Filter):
    """Tag every line logged during a turn (cache, guards, rerank, SDK retries ...) with that turn's request_id
    and thread_id, read from the graph config runner.py starts the turn with. Lines logged outside a graph run
    (ingest, eval, the UI) are left as they are; ids a line already carries are kept."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from langgraph.config import get_config
            conf = get_config().get("configurable", {})
        except Exception:                            # outside a graph run (RuntimeError), or langgraph missing
            return True
        ids = {k: conf[k] for k in ("request_id", "thread_id") if conf.get(k)}
        if ids:
            record.json_fields = {**ids, **(getattr(record, "json_fields", None) or {})}
        return True


def setup_logging(app: str) -> logging.Logger:
    """Configure the 'ml_researcher' logger once per process; safe to call on every Streamlit rerun."""
    logger = logging.getLogger(LOGGER)
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False                         # don't duplicate into Streamlit's/root handlers

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(TurnIds())
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)                # keep the terminal / UI quiet
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)

    if LOG_TO_GCP:
        try:
            import google.cloud.logging
            from google.cloud.logging.handlers import CloudLoggingHandler
            client = google.cloud.logging.Client(project=PROJECT)
            cloud = CloudLoggingHandler(client, name=LOG_NAME, labels={"app": app})
            cloud.addFilter(TurnIds())
            logger.addHandler(cloud)
        except Exception as e:                       # never let logging break the app
            logger.warning("Cloud Logging disabled: %s", e)

    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "google.auth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # the Gemini SDK logs each automatic retry ("Retrying ... as it raised ReadTimeout / 429") at INFO:
    # keep those in our file so timeouts and rate limits leave a trace
    retries = logging.getLogger("google_genai._api_client")
    retries.setLevel(logging.INFO)
    retries.addHandler(file_handler)
    retries.propagate = False
    logger._configured = True
    return logger
