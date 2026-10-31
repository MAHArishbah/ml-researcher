"""One pacer for every model call in this process: chat models (any provider), embeddings, captions, caches.

Calls are spaced evenly at `rps` (no bursts), across all threads. Interactive use runs at LLM_RPS;
batch jobs (ingest, evaluate) call batch_mode() to drop to BATCH_RPS, so long runs look like steady,
modest traffic instead of spikes of thousands of requests.

On top of each process's own pace, ALL processes (UI, chat, a scheduled ingest, an eval job) share one
combined limit of LLM_RPS calls per second, coordinated through a locked file. So running the UI while
the daily ingest works never adds up to more than LLM_RPS against the project.
"""
import asyncio
import fcntl
import logging
import os
import threading
import time
from pathlib import Path

from langchain_core.rate_limiters import BaseRateLimiter

from .config import BATCH_RPS, LLM_RPS

log = logging.getLogger("ml_researcher.ratelimit")
SHARED = Path(os.path.expanduser("~/.ml_researcher/pacer.slot"))   # next free slot (wall-clock s), all processes
_shared_lock = threading.Lock()


def _reserve_shared(rps: float) -> float:
    """Book the next slot on the cross-process timeline (one slot per 1/rps s); return the wait.
    Falls back to no extra wait if the file can't be used, so pacing never breaks a call.Coordinates seprate processes"""
    try:
        with _shared_lock:
            SHARED.parent.mkdir(parents=True, exist_ok=True)
            with open(SHARED, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)              # other processes wait here
                try:
                    f.seek(0)
                    raw = f.read().strip()
                    now = time.time()
                    start = max(now, float(raw) if raw else 0.0)
                    f.seek(0)
                    f.truncate()
                    f.write(repr(start + 1 / rps))
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        return start - now
    except (OSError, ValueError) as e:
        log.warning("shared rate limit unavailable, using per-process pace only: %s", e)
        return 0.0


class Pacer(BaseRateLimiter):
    def __init__(self, rps: float):
        self._lock = threading.Lock()
        self._next = 0.0                         # monotonic time of the next free slot
        self.rps = rps if rps > 0 else None

    def set_rate(self, rps: float) -> None:
        with self._lock:
            self.rps = rps if rps > 0 else None

    def _reserve(self) -> float:
        """Book the next slot; return how long the caller must wait for it. For one process, coordinates threads with in one process."""
        with self._lock:
            if not self.rps:
                return 0.0
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + 1 / self.rps
            return start - now

    def acquire(self, *, blocking: bool = True) -> bool:     # used by LangChain chat models
        wait = self._reserve()                                # this process's own pace
        if wait > 0:
            time.sleep(wait)
        if LLM_RPS > 0:                                       # then the combined limit across processes
            wait = _reserve_shared(LLM_RPS)
            if wait > 0:
                time.sleep(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)
        if LLM_RPS > 0:
            wait = _reserve_shared(LLM_RPS)
            if wait > 0:
                await asyncio.sleep(wait)
        return True


PACER = Pacer(LLM_RPS)


def wait() -> None:
    """Call before every direct google-genai request (LangChain models use PACER themselves)."""
    PACER.acquire()


def batch_mode() -> None:
    PACER.set_rate(BATCH_RPS)
    log.info("batch mode: model calls paced at %s/s", BATCH_RPS)
    print(f"batch mode: model calls limited to {BATCH_RPS}/s")
