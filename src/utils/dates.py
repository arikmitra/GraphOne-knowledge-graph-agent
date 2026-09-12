"""
Date normalization for the Phase II "Freshness Challenge".

Handles three cases:
1. Structured dates (ISO-8601, RFC-822, common meta tags) -> parsed directly.
2. Relative dates ("2 hours ago", "yesterday", "3 days ago") -> resolved
   against a reference "now" (injectable for testing).
3. No date at all -> heuristic fallback using a content hash + a seen-log,
   so a source with no timestamp is still classified fresh/stale by
   "have we seen this exact content before", not by guessing a date.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from dateutil import parser as dateparser

RELATIVE_PATTERN = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>second|sec|minute|min|hour|hr|day|week|month)s?\s*ago",
    re.IGNORECASE,
)

UNIT_TO_TIMEDELTA = {
    "second": lambda n: timedelta(seconds=n),
    "sec": lambda n: timedelta(seconds=n),
    "minute": lambda n: timedelta(minutes=n),
    "min": lambda n: timedelta(minutes=n),
    "hour": lambda n: timedelta(hours=n),
    "hr": lambda n: timedelta(hours=n),
    "day": lambda n: timedelta(days=n),
    "week": lambda n: timedelta(weeks=n),
    "month": lambda n: timedelta(days=30 * n),
}

WORD_DATES = {
    "just now": timedelta(seconds=0),
    "today": timedelta(hours=0),
    "yesterday": timedelta(days=1),
}


def parse_relative(text: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse strings like '2 hours ago' / 'yesterday' relative to `now`."""
    now = now or datetime.now(timezone.utc)
    text_norm = text.strip().lower()

    if text_norm in WORD_DATES:
        return now - WORD_DATES[text_norm]

    m = RELATIVE_PATTERN.search(text_norm)
    if m:
        num = int(m.group("num"))
        unit = m.group("unit")
        return now - UNIT_TO_TIMEDELTA[unit](num)

    return None


def normalize_date(
    raw: Optional[str],
    now: Optional[datetime] = None,
) -> tuple[Optional[datetime], str]:
    """
    Best-effort normalization of any date string we scraped.

    Returns (parsed_datetime_or_None, method) where method is one of:
    "iso", "relative", "dateutil", "unparseable".
    """
    now = now or datetime.now(timezone.utc)
    if not raw or not raw.strip():
        return None, "unparseable"

    raw = raw.strip()

    # Fast path: already ISO-8601.
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, "iso"
    except ValueError:
        pass

    rel = parse_relative(raw, now)
    if rel is not None:
        return rel, "relative"

    # General fallback via dateutil, which handles most RFC-822 / natural
    # language formats found in RSS feeds and <meta> tags.
    try:
        dt = dateparser.parse(raw, fuzzy=True)
        if dt is None:
            return None, "unparseable"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, "dateutil"
    except (ValueError, OverflowError):
        return None, "unparseable"


def is_fresh(dt: Optional[datetime], now: Optional[datetime] = None, window_hours: int = 24) -> bool:
    if dt is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - dt) <= timedelta(hours=window_hours)


def content_fingerprint(text: str) -> str:
    """Stable hash used for the no-date heuristic and dedup-across-runs."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SeenStore:
    """
    Minimal seen-content tracker for the "no strict date" heuristic and for
    cross-run dedup (Phase VI freshness-tracking requirement). Backed by a
    plain set here; in production this is a Redis SET or a Postgres table
    keyed by content_fingerprint with a TTL (see architecture.md).
    """

    def __init__(self):
        self._seen: set[str] = set()

    def is_new(self, text: str) -> bool:
        fp = content_fingerprint(text)
        if fp in self._seen:
            return False
        self._seen.add(fp)
        return True
