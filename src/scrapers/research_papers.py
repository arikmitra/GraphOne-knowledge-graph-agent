"""
Phase I research papers vertical: Arxiv + GitHub star correlation.

Uses Arxiv's official Atom export API (bulk-friendly, no anti-bot concerns,
respects their documented rate limits) rather than scraping arxiv.org HTML
directly -- this is both more reliable and better etiquette for a
"hundreds of thousands of records" style workload. GitHub stars are pulled
from the public REST API (unauthenticated: 60 req/hr; with a token:
5000 req/hr -- the client below batches accordingly).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import feedparser

from src.utils.http import HttpClient, DomainPolicy

logger = logging.getLogger("graphone.scrapers.papers")

ARXIV_API = "http://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com"


async def fetch_arxiv_batch(
    client: HttpClient,
    query: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
    start: int = 0,
    max_results: int = 100,
) -> list[dict]:
    """
    One page of Arxiv results. Called in a loop with increasing `start`
    to page through to the target volume -- this is the mechanism that
    lets the same code scale from 100 to 500,000 records: only the loop
    bound changes, not the code (Phase I scalability requirement).
    """
    url = (
        f"{ARXIV_API}?search_query={quote(query)}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    result = await client.get(url)
    if result is None:
        return []
    status, body, _ = result
    if status != 200:
        logger.warning("arxiv page start=%s returned %s", start, status)
        return []

    feed = feedparser.parse(body)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1] if "/abs/" in entry.id else entry.id
        papers.append({
            "title": re.sub(r"\s+", " ", entry.title).strip(),
            "authors": [a.name for a in getattr(entry, "authors", [])],
            "paper_url": entry.id,
            "published_date": entry.get("published"),
            "abstract": re.sub(r"\s+", " ", entry.get("summary", "")).strip(),
            "arxiv_id": arxiv_id,
            "source_name": "arxiv.org",
        })
    return papers


GITHUB_URL_PATTERN = re.compile(r"github\.com/([\w.-]+/[\w.-]+)")


def extract_github_repo(text: str) -> Optional[str]:
    """Find a github.com/owner/repo reference inside an abstract or comment field."""
    m = GITHUB_URL_PATTERN.search(text or "")
    if not m:
        return None
    return m.group(1).rstrip(".,)")


async def fetch_github_stars(client: HttpClient, owner_repo: str, token: Optional[str] = None) -> Optional[int]:
    """Fetch current star count for a repo, respecting GitHub's rate limits."""
    url = f"{GITHUB_API}/repos/{owner_repo}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    result = await client.get(url, headers=headers)
    if result is None:
        return None
    status, body, _ = result
    if status != 200:
        return None
    import json
    try:
        data = json.loads(body)
        return data.get("stargazers_count")
    except json.JSONDecodeError:
        return None


async def scrape_papers_with_stars(
    client: HttpClient,
    target_count: int = 1000,
    page_size: int = 100,
    github_token: Optional[str] = None,
) -> list[dict]:
    """
    Full Phase I research-papers flow: page through Arxiv, then for every
    paper whose abstract references a GitHub repo, correlate current star
    count concurrently (bounded by the GitHub domain policy).
    """
    client.set_policy("export.arxiv.org", DomainPolicy(max_concurrency=1, requests_per_second=0.33))
    client.set_policy("api.github.com", DomainPolicy(max_concurrency=10, requests_per_second=1.0))

    papers: list[dict] = []
    start = 0
    while len(papers) < target_count:
        batch = await fetch_arxiv_batch(client, start=start, max_results=page_size)
        if not batch:
            break
        papers.extend(batch)
        start += page_size
        await asyncio.sleep(3.1)  # respect Arxiv's documented ~1 req/3s guidance

    papers = papers[:target_count]

    async def enrich(paper: dict) -> dict:
        repo = extract_github_repo(paper.get("abstract", ""))
        if repo:
            stars = await fetch_github_stars(client, repo, token=github_token)
            paper["github_url"] = f"https://github.com/{repo}"
            paper["github_stars"] = stars
        else:
            paper["github_url"] = None
            paper["github_stars"] = None
        return paper

    enriched = await asyncio.gather(*(enrich(p) for p in papers))
    return list(enriched)
