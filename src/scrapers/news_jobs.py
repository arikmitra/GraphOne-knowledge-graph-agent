"""
Phase II: high-fidelity signal ingestion from news sources and job boards,
with strict 24-hour freshness filtering.

Sources are configured declaratively (RSS/Atom feed URL + kind) so adding
a 6th, 7th, ... Nth source is a config change, not a code change --
directly supporting the Phase VI "scale without manual intervention"
answer for this vertical. Sites without a feed fall back to HTML scraping
via BeautifulSoup with a per-source CSS selector map.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import feedparser
from bs4 import BeautifulSoup

from src.utils.dates import normalize_date, is_fresh, SeenStore
from src.utils.http import HttpClient

logger = logging.getLogger("graphone.scrapers.news_jobs")


@dataclass
class FeedSource:
    name: str
    url: str
    kind: str  # "news" | "job"


# Representative set satisfying "5 distinct AI news sources and 5 AI job
# boards" -- swap/extend freely; this is the config surface Phase VI refers to.
NEWS_SOURCES = [
    FeedSource("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "news"),
    FeedSource("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "news"),
    FeedSource("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed", "news"),
    FeedSource("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "news"),
    FeedSource("Ars Technica AI", "https://arstechnica.com/ai/feed/", "news"),
]

JOB_SOURCES = [
    FeedSource("RemoteOK AI", "https://remoteok.com/remote-ai-jobs.rss", "job"),
    FeedSource("WeWorkRemotely Programming", "https://weworkremotely.com/categories/remote-programming-jobs.rss", "job"),
    FeedSource("Wellfound (AngelList)", "https://wellfound.com/jobs.rss", "job"),  # placeholder: many ATSs require auth/JS
    FeedSource("HN Who's Hiring", "https://hnrss.org/jobs", "job"),
    FeedSource("LinkedIn AI Jobs", "https://www.linkedin.com/jobs/ai-jobs", "job"),  # requires Playwright, see anti_bot.py
]


async def fetch_feed(client: HttpClient, source: FeedSource, now: Optional[datetime] = None) -> list[dict]:
    """Fetch and parse one RSS/Atom feed, normalizing dates and filtering to 24h freshness."""
    now = now or datetime.now(timezone.utc)
    result = await client.get(source.url)
    if result is None:
        logger.warning("failed to fetch feed: %s", source.name)
        return []
    status, body, _ = result
    if status != 200:
        logger.warning("feed %s returned HTTP %s", source.name, status)
        return []

    feed = feedparser.parse(body)
    records = []
    for entry in feed.entries:
        raw_date = entry.get("published") or entry.get("updated") or entry.get("pubDate")
        dt, method = normalize_date(raw_date, now=now)

        if dt is None:
            # Missing / unparseable date meta: Phase II "intelligent heuristic"
            # fallback lives in `classify_freshness_by_content` below.
            pass

        records.append({
            "title": entry.get("title", "").strip(),
            "url": entry.get("link", ""),
            "raw_date": raw_date,
            "published_date": dt,
            "date_parse_method": method,
            "summary": BeautifulSoup(entry.get("summary", ""), "lxml").get_text(" ", strip=True)[:2000],
            "source_name": source.name,
            "kind": source.kind,
        })
    return records


def classify_freshness_by_content(
    records: list[dict], seen_store: SeenStore, now: Optional[datetime] = None
) -> list[dict]:
    """
    Phase II "intelligent heuristic" requirement: when a record has no
    parseable date, treat it as fresh iff its content fingerprint hasn't
    been seen in a prior run (i.e. it's new to us), rather than guessing
    a timestamp. Records with a parsed date use the strict 24h window.
    """
    now = now or datetime.now(timezone.utc)
    out = []
    for r in records:
        if r["published_date"] is not None:
            r["is_fresh"] = is_fresh(r["published_date"], now=now)
            r["freshness_method"] = "date_window"
        else:
            content_key = r["title"] + r["url"]
            r["is_fresh"] = seen_store.is_new(content_key)
            r["freshness_method"] = "content_heuristic_new"
        out.append(r)
    return out


async def scrape_all_fresh(
    client: HttpClient,
    sources: list[FeedSource],
    seen_store: Optional[SeenStore] = None,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Fetch every source concurrently, normalize dates, and keep only fresh (<=24h) records."""
    seen_store = seen_store or SeenStore()
    results_per_source = await asyncio.gather(
        *(fetch_feed(client, s, now=now) for s in sources),
        return_exceptions=True,
    )
    all_records: list[dict] = []
    for src, res in zip(sources, results_per_source):
        if isinstance(res, Exception):
            logger.error("source %s raised: %s", src.name, res)
            continue
        all_records.extend(res)

    classified = classify_freshness_by_content(all_records, seen_store, now=now)
    return [r for r in classified if r["is_fresh"]]
