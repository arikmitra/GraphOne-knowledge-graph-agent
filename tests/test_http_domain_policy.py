import asyncio
import time
 
import pytest
 
from src.utils.http import DomainPolicy
 
 
def test_sub_one_rate_does_not_raise():
    """
    Regression test: DomainPolicy(requests_per_second=0.33) (the Arxiv policy)
    used to raise `ValueError: Amount must be a number between 0 and the
    maximum capacity` from aiolimiter, because AsyncLimiter(0.33, ...) sets
    bucket capacity to 0.33 while acquire() requests amount=1 by default.
    """
    policy = DomainPolicy(max_concurrency=1, requests_per_second=0.33)
    assert policy._limiter.max_rate >= 1
 
 
def test_normal_rate_unaffected():
    policy = DomainPolicy(max_concurrency=5, requests_per_second=5.0)
    assert policy._limiter.max_rate == 5.0
    assert policy._limiter.time_period == 1.0
 
 
@pytest.mark.asyncio
async def test_sub_one_rate_actually_throttles():
    """0.33 req/s should space consecutive acquires by ~1/0.33 = 3.03s."""
    policy = DomainPolicy(max_concurrency=1, requests_per_second=0.33)
    t0 = time.monotonic()
    async with policy._semaphore:
        async with policy._limiter:
            pass
    first = time.monotonic() - t0
    async with policy._semaphore:
        async with policy._limiter:
            pass
    second = time.monotonic() - t0
 
    assert first < 0.5  # bucket starts full, first acquire is immediate
    assert second == pytest.approx(3.03, abs=0.5)
 
 
@pytest.mark.asyncio
async def test_very_low_rate_does_not_raise():
    """Even a very low rate (0.01 req/s) must not raise, just throttle harder."""
    policy = DomainPolicy(max_concurrency=1, requests_per_second=0.01)
    async with policy._semaphore:
        async with policy._limiter:
            pass  # should not raise