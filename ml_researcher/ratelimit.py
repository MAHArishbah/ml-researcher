"""One pacer for every model call in this process: chat models (any provider), embeddings, captions, caches.

Calls are spaced evenly at `rps` (no bursts), across all threads. Interactive use runs at LLM_RPS;
batch jobs (ingest, evaluate) call batch_mode() to drop to BATCH_RPS, so long runs look like steady,
modest traffic instead of spikes of thousands of requests.
"""
import asyncio
import logging
import threading
import time

from langchain_core.rate_limiters import BaseRateLimiter

from .config import BATCH_RPS, LLM_RPS

log = logging.getLogger("ml_researcher.ratelimit")


class Pacer(BaseRateLimiter):
    def __init__(self, rps: float):
        self._lock = threading.Lock()
        self._next = 0.0                         # monotonic time of the next free slot
        self.rps = rps if rps > 0 else None

    def set_rate(self, rps: float) -> None:
        with self._lock:
            self.rps = rps if rps > 0 else None

    def _reserve(self) -> float:
        """Book the next slot; return how long the caller must wait for it."""
        with self._lock:
            if not self.rps:
                return 0.0
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + 1 / self.rps
            return start - now

    def acquire(self, *, blocking: bool = True) -> bool:     # used by LangChain chat models
        wait = self._reserve()
        if wait > 0:
            time.sleep(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        wait = self._reserve()
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
