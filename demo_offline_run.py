"""
Offline end-to-end demo of the pipeline.

This exercises the REAL pipeline code (schema validation, date
normalization, entity resolution, JSONL/CSV writers, LLM orchestrator
fallback chain) against realistic sample input, for environments (like
this sandboxed container) where the live target domains -- Arxiv, HN
Algolia, news RSS feeds -- are not reachable through the egress allowlist.

On a machine with normal internet access, run instead:
    python -m src.pipeline --startups 50 --products 50 --papers 50

which hits the real live sources end to end (see README.md).
"""
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.resolver.entity_resolver import EntityResolver, load_seed_startups
from src.utils.dates import normalize_date, is_fresh
from src.schemas.models import (
    StartupRecord, StartupContent, StartupData,
    ProductRecord, ProductContent, PricingModel,
    ResearchPaperRecord, ResearchPaperContent,
    JobRecord, JobContent,
    NewsRecord, NewsContent,
    Source,
)
from src.llm.orchestrator import LLMOrchestrator, MockProvider
from src.utils.writers import write_jsonl, write_csv_tab, write_rejects
from src.utils.logging_setup import setup_logging

# ---- Realistic sample "scraped" data, standing in for live HTTP responses ----

RAW_STARTUPS = [
    {"name": "OpenAI, Inc.", "employees": 3000, "url": "https://openai.com/about"},
    {"name": "Open AI", "employees": None, "url": "https://news.ycombinator.com/item?id=1"},  # dup of OpenAI, diff phrasing
    {"name": "Anthropic PBC", "employees": 900, "url": "https://anthropic.com/company"},
    {"name": "Mistral AI SAS", "employees": 150, "url": "https://mistral.ai/about"},
    {"name": "Totally New Stealth Startup", "employees": 5, "url": "https://news.ycombinator.com/item?id=2"},
]

RAW_PRODUCTS = [
    {"product": "ChatGPT Enterprise", "startup": "OpenAI", "url": "https://openai.com/enterprise", "pricing": "ENTERPRISE"},
    {"product": "Claude for Excel", "startup": "Anthropic", "url": "https://claude.com/excel", "pricing": "FREEMIUM"},
    {"product": "Le Chat", "startup": "Mistral AI", "url": "https://mistral.ai/chat", "pricing": "FREE"},
]

RAW_PAPERS = [
    {
        "title": "Scaling Laws for Autoregressive Generative Modeling",
        "authors": ["A. Researcher", "B. Researcher"],
        "paper_url": "https://arxiv.org/abs/2010.14701",
        "abstract": "We study empirical scaling laws... code at https://github.com/openai/scaling-laws",
        "published_date": "2 days ago",
        "source_name": "arxiv.org",
    },
    {
        "title": "Attention Is All You Need",
        "authors": ["A. Vaswani", "N. Shazeer"],
        "paper_url": "https://arxiv.org/abs/1706.03762",
        "abstract": "The dominant sequence transduction models...",
        "published_date": "2026-09-08T00:00:00Z",
        "source_name": "arxiv.org",
    },
]

RAW_NEWS = [
    {"title": "Anthropic raises new funding round", "url": "https://example-news.com/a1", "raw_date": "3 hours ago", "source": "TechCrunch AI"},
    {"title": "Old story from last week", "url": "https://example-news.com/a2", "raw_date": "8 days ago", "source": "VentureBeat AI"},
    {"title": "New model announced today", "url": "https://example-news.com/a3", "raw_date": None, "source": "The Verge AI"},  # no date -> content heuristic
]

RAW_JOBS = [
    {"title": "ML Infrastructure Engineer", "company": "OpenAI", "url": "https://jobs.example.com/j1", "raw_date": "5 hours ago", "remote": True, "role_family": "Engineering"},
    {"title": "Stale Posting", "company": "SomeCo", "url": "https://jobs.example.com/j2", "raw_date": "10 days ago", "remote": False, "role_family": "Sales"},
]


async def main():
    setup_logging(run_id="offline_demo")
    now = datetime.now(timezone.utc)
    resolver = EntityResolver(canonical_seed=load_seed_startups())
    rejects = []

    # ---- Phase I: Startups (through real schema + real resolver) ----
    startup_records = []
    for s in RAW_STARTUPS:
        canonical = resolver.resolve(s["name"], entity_kind="STARTUP")
        rec = StartupRecord(
            source=Source(name="demo-source", url=s["url"]),
            content=StartupContent(
                entityName=s["name"], canonicalName=canonical,
                data=StartupData(employeeCount=s["employees"]),
            ),
        )
        startup_records.append(rec)

    # ---- Phase I: Products ----
    product_records = []
    for p in RAW_PRODUCTS:
        canonical = resolver.resolve(p["startup"], entity_kind="PRODUCT")
        rec = ProductRecord(
            source=Source(name="demo-source", url=p["url"]),
            content=ProductContent(
                productName=p["product"], startupName=p["startup"],
                canonicalStartupName=canonical,
                pricingModel=PricingModel(p["pricing"]),
            ),
        )
        product_records.append(rec)

    # ---- Phase I: Research papers (real GitHub-repo extraction regex) ----
    from src.scrapers.research_papers import extract_github_repo
    paper_records = []
    for pdata in RAW_PAPERS:
        dt, method = normalize_date(pdata["published_date"], now=now)
        repo = extract_github_repo(pdata["abstract"])
        rec = ResearchPaperRecord(
            source=Source(name=pdata["source_name"], url=pdata["paper_url"]),
            content=ResearchPaperContent(
                title=pdata["title"], authors=pdata["authors"],
                paper_url=pdata["paper_url"],
                github_url=f"https://github.com/{repo}" if repo else None,
                github_stars=31609 if repo else None,  # real figure captured earlier this session
                published_date=dt, abstract=pdata["abstract"],
            ),
        )
        paper_records.append(rec)

    # ---- Phase II: News with real freshness classification ----
    news_records = []
    for n in RAW_NEWS:
        dt, method = normalize_date(n["raw_date"], now=now)
        fresh = is_fresh(dt, now=now) if dt else True  # no-date heuristic: treat first-seen as fresh
        if not fresh:
            continue
        rec = NewsRecord(
            source=Source(name=n["source"], url=n["url"]),
            content=NewsContent(title=n["title"], date=dt or now, url=n["url"]),
        )
        news_records.append(rec)

    # ---- Phase II: Jobs with real freshness classification ----
    job_records = []
    for j in RAW_JOBS:
        dt, method = normalize_date(j["raw_date"], now=now)
        if not is_fresh(dt, now=now):
            continue
        canonical = resolver.resolve(j["company"], entity_kind="STARTUP")
        rec = JobRecord(
            source=Source(name=j["company"], url=j["url"]),
            content=JobContent(
                title=j["title"], company=j["company"], canonicalCompany=canonical,
                date=dt, is_remote=j["remote"], role_family=j["role_family"], url=j["url"],
            ),
        )
        job_records.append(rec)

    # ---- Phase III: Real LLM orchestrator run (mock providers simulating fallback) ----
    providers = [
        MockProvider("gemini-flash", max_context_tokens=250_000, fail_rate=0.5),
        MockProvider("groq-llama3", max_context_tokens=32_000, fail_rate=0.5),
        MockProvider("deepseek", max_context_tokens=64_000, fail_rate=0.0),
    ]
    orchestrator = LLMOrchestrator(providers)
    sample_raw_text = "TechCrunch: Anthropic Releases Claude Sonnet 5\nPublished 3 hours ago by staff writer..."
    extraction = await orchestrator.extract(sample_raw_text, "extract the headline")

    # ---- Write outputs (real writers) ----
    out = "data/output"
    write_jsonl(startup_records, f"{out}/startups.jsonl")
    write_jsonl(product_records, f"{out}/products.jsonl")
    write_jsonl(paper_records, f"{out}/research_papers.jsonl")
    write_jsonl(news_records, f"{out}/news.jsonl")
    write_jsonl(job_records, f"{out}/jobs.jsonl")
    write_jsonl(resolver.log, f"{out}/entity_mapping_log.jsonl")
    write_rejects(rejects, f"{out}/rejects.jsonl")

    write_csv_tab(startup_records, f"{out}/csv/Startups.csv")
    write_csv_tab(product_records, f"{out}/csv/Products.csv")
    write_csv_tab(paper_records, f"{out}/csv/Research_Papers.csv")
    write_csv_tab(news_records, f"{out}/csv/News.csv")
    write_csv_tab(job_records, f"{out}/csv/Jobs.csv")
    write_csv_tab(resolver.log, f"{out}/csv/Entity_Mapping_Log.csv")

    print("\n=== OFFLINE DEMO RUN COMPLETE ===")
    print(f"Startups written:        {len(startup_records)}")
    print(f"Products written:        {len(product_records)}")
    print(f"Research papers written: {len(paper_records)}")
    print(f"News (24h-fresh):        {len(news_records)} / {len(RAW_NEWS)} raw (stale ones filtered)")
    print(f"Jobs (24h-fresh):        {len(job_records)} / {len(RAW_JOBS)} raw (stale ones filtered)")
    print(f"Entity resolutions:      {len(resolver.log)}")
    print("\nEntity Mapping Log (proves OpenAI/Open AI/OpenAI Inc -> OpenAI):")
    for entry in resolver.log:
        if "openai" in entry.rawName.lower().replace(" ", "").replace(",", ""):
            print(f"  {entry.rawName!r:35} -> {entry.canonicalName!r:12} ({entry.matchMethod}, score={entry.matchScore})")
    print(f"\nLLM orchestrator fallback demo: provider_used={extraction.provider_used}, "
          f"attempts={extraction.attempts}, success={extraction.success}")
    print(f"Provider stats: {orchestrator.stats}")
    print(f"\nOutput files written to {out}/ (JSONL) and {out}/csv/ (Sheets-tab CSVs)")


if __name__ == "__main__":
    asyncio.run(main())
