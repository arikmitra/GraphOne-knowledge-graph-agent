"""
Shared async HTTP layer.

Design notes (relevant to Phase I / V requirements):
- Per-domain concurrency + rate limiting so one slow/strict domain never
  starves the others and we don't hammer a single host into a hard block.
- Exponential backoff with full jitter on 429/5xx, honoring Retry-After
  when the server sends one.
- A rotating pool of realistic desktop User-Agents + baseline headers to
  look like a normal browser rather than python-requests/aiohttp defaults.
- Every failure is logged with enough context (url, status, attempt) to
  debug post-hoc; nothing fails silently.
"""
from __future__ import annotations
 
import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional
 
import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)
 
logger = logging.getLogger("graphone.http")
 
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
]
 
 
class RetryableStatus(Exception):
    """Raised for 429/5xx responses so tenacity can retry the call."""
    def __init__(self, status: int, retry_after: Optional[float] = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"retryable HTTP {status}")
 
 
@dataclass
class DomainPolicy:
    """Per-domain crawl policy -- concurrency + requests/sec budget."""
    max_concurrency: int = 5
    requests_per_second: float = 2.0
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _limiter: AsyncLimiter = field(init=False, repr=False)
 
    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        rate = max(self.requests_per_second, 0.001)
        if rate >= 1:
            # e.g. 5 req/s -> max_rate=5, time_period=1s
            self._limiter = AsyncLimiter(rate, time_period=1.0)
        else:
            # Sub-1 rates (e.g. 0.33 req/s = 1 request per 3s) can't be expressed
            # as a fractional bucket capacity -- AsyncLimiter.acquire() defaults to
            # requesting amount=1, which must be <= max_rate. So instead we keep
            # max_rate=1 and stretch time_period to 1/rate seconds, giving the same
            # effective throughput without violating the capacity constraint.
            self._limiter = AsyncLimiter(1, time_period=1.0 / rate)
 
 
class HttpClient:
    """
    Wraps an aiohttp.ClientSession with per-domain policies, retry/backoff,
    and header rotation. One instance is shared across a pipeline run.
    """
 
    def __init__(self, default_policy: Optional[DomainPolicy] = None, timeout_s: int = 30):
        self._session: Optional[aiohttp.ClientSession] = None
        self._policies: dict[str, DomainPolicy] = {}
        self._default_policy = default_policy or DomainPolicy()
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "429s": 0}
 
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self
 
    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()
 
    def set_policy(self, domain: str, policy: DomainPolicy):
        self._policies[domain] = policy
 
    def _policy_for(self, domain: str) -> DomainPolicy:
        return self._policies.get(domain, self._default_policy)
 
    @staticmethod
    def _domain(url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc
 
    def _headers(self) -> dict:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }
 
    @retry(
        retry=retry_if_exception_type(RetryableStatus),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_with_retry(self, url: str, headers: Optional[dict] = None, **kwargs) -> tuple[int, str, dict]:
        assert self._session is not None, "use HttpClient as an async context manager"
        self.stats["requests"] += 1
        merged_headers = {**self._headers(), **(headers or {})}
        async with self._session.get(url, headers=merged_headers, **kwargs) as resp:
            if resp.status == 429:
                self.stats["429s"] += 1
                self.stats["retries"] += 1
                retry_after = resp.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else None
                if wait_s:
                    await asyncio.sleep(min(wait_s, 60))
                raise RetryableStatus(429, wait_s)
            if resp.status >= 500:
                self.stats["retries"] += 1
                raise RetryableStatus(resp.status)
            body = await resp.text(errors="replace")
            return resp.status, body, dict(resp.headers)
 
    async def get(self, url: str, **kwargs) -> Optional[tuple[int, str, dict]]:
        """Fetch a URL respecting the domain's concurrency + rate policy."""
        domain = self._domain(url)
        policy = self._policy_for(domain)
        async with policy._semaphore:
            async with policy._limiter:
                try:
                    return await self._get_with_retry(url, **kwargs)
                except Exception as e:
                    self.stats["failures"] += 1
                    logger.error("giving up on %s after retries: %s", url, e)
                    return None