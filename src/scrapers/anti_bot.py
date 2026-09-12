"""
Phase V: Anti-Bot & Scale Thinking.

For Cloudflare / Datadome-protected or heavily JS-rendered targets, plain
aiohttp GETs get blocked or served an empty shell. This module uses
Playwright's async API (headless Chromium) with a set of measures that
meaningfully reduce automated-traffic fingerprinting:

  - Persistent browser context reused across pages (avoids the
    all-headless-launches-look-identical fingerprint of a fresh browser
    per request).
  - navigator.webdriver patched out, plausible viewport/locale/timezone,
    and realistic Accept-Language / Sec-CH-UA headers.
  - Randomized human-like delays between actions instead of instant
    navigation-then-scrape.
  - Waiting for network-idle / specific selectors rather than a fixed
    sleep, so we don't scrape a still-loading challenge page.
  - A concurrency cap much lower than the plain-HTTP scraper's, because
    each browser context is expensive and aggressive parallelism is
    itself a detection signal.

This is NOT a captcha-solving or Cloudflare-bypass-service integration --
per the assignment scope ("demonstrate OR comprehensively document your
strategy"), the documented strategy for sources that still block this
approach is in architecture.md: residential proxy rotation + managed
unblocking APIs (e.g. Bright Data, ScraperAPI, Zyte) as the production
answer for the hardest targets, rather than trying to defeat Cloudflare's
managed challenge directly, which is both unreliable and increasingly a
ToS/legal grey area to build custom in-house.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger("graphone.scrapers.anti_bot")

STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
window.chrome = { runtime: {} };
"""


class StealthBrowserPool:
    """
    Manages a small pool of persistent, stealth-configured browser contexts
    for JS-heavy / bot-protected targets. Kept separate from the plain
    HttpClient so the cheap bulk-HTTP path (Arxiv, RSS feeds, GitHub API)
    never pays the cost of spinning up a browser.
    """

    def __init__(self, pool_size: int = 3, headless: bool = True):
        self.pool_size = pool_size
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._contexts: list[BrowserContext] = []
        self._semaphore = asyncio.Semaphore(pool_size)

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        for _ in range(self.pool_size):
            ctx = await self._browser.new_context(
                viewport={"width": 1366, "height": 768},
                locale="en-US",
                timezone_id="America/New_York",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                ),
            )
            await ctx.add_init_script(STEALTH_INIT_SCRIPT)
            self._contexts.append(ctx)
        return self

    async def __aexit__(self, *exc):
        for ctx in self._contexts:
            await ctx.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def fetch_rendered(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        wait_until: str = "networkidle",
        max_retries: int = 3,
    ) -> Optional[str]:
        """Load a URL in a pooled stealth context and return the rendered HTML."""
        async with self._semaphore:
            ctx = random.choice(self._contexts)
            page = await ctx.new_page()
            try:
                for attempt in range(1, max_retries + 1):
                    try:
                        await asyncio.sleep(random.uniform(0.5, 2.0))  # human-ish pacing
                        await page.goto(url, wait_until=wait_until, timeout=30_000)
                        if wait_selector:
                            await page.wait_for_selector(wait_selector, timeout=15_000)
                        # Small human-like scroll to trigger lazy content / look less scripted.
                        await page.mouse.wheel(0, random.randint(200, 800))
                        await asyncio.sleep(random.uniform(0.3, 1.0))
                        return await page.content()
                    except Exception as e:
                        logger.warning("attempt %s/%s failed for %s: %s", attempt, max_retries, url, e)
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                return None
            finally:
                await page.close()
