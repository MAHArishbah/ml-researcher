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
            logger.addHandler(CloudLoggingHandler(client, name=LOG_NAME, labels={"app": app}))
        except Exception as e:                       # never let logging break the app
            logger.warning("Cloud Logging disabled: %s", e)

    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "google.auth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger._configured = True
    return logger
