from datetime import datetime, timezone, timedelta

from src.utils.dates import normalize_date, parse_relative, is_fresh, SeenStore, content_fingerprint

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def test_iso_date():
    dt, method = normalize_date("2026-09-09T08:30:00Z", now=NOW)
    assert method == "iso"
    assert dt.year == 2026 and dt.month == 9 and dt.day == 9


def test_relative_hours_ago():
    dt = parse_relative("2 hours ago", now=NOW)
    assert dt == NOW - timedelta(hours=2)


def test_relative_yesterday():
    dt = parse_relative("Yesterday", now=NOW)
    assert dt == NOW - timedelta(days=1)


def test_relative_minutes():
    dt = parse_relative("45 minutes ago", now=NOW)
    assert dt == NOW - timedelta(minutes=45)


def test_dateutil_fallback_rfc822():
    dt, method = normalize_date("Wed, 09 Sep 2026 10:00:00 GMT", now=NOW)
    assert method in ("dateutil", "iso")
    assert dt.day == 9


def test_unparseable():
    dt, method = normalize_date("not a date at all !!!", now=NOW)
    assert dt is None
    assert method == "unparseable"


def test_missing_date():
    dt, method = normalize_date(None, now=NOW)
    assert dt is None
    assert method == "unparseable"


def test_is_fresh_within_window():
    dt = NOW - timedelta(hours=5)
    assert is_fresh(dt, now=NOW, window_hours=24) is True


def test_is_fresh_outside_window():
    dt = NOW - timedelta(hours=30)
    assert is_fresh(dt, now=NOW, window_hours=24) is False


def test_is_fresh_none():
    assert is_fresh(None, now=NOW) is False


def test_seen_store_dedup():
    store = SeenStore()
    assert store.is_new("Article A" + "http://x.com/a") is True
    assert store.is_new("Article A" + "http://x.com/a") is False
    assert store.is_new("Article B" + "http://x.com/b") is True


def test_content_fingerprint_normalizes_whitespace():
    a = content_fingerprint("Hello   World")
    b = content_fingerprint("hello world")
    assert a == b
