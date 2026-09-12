"""
End-to-end pipeline runner tying together Phases I-IV.
 
Usage:
    python -m src.pipeline --papers 50 --startups 50 --news --jobs \
        --output-dir data/output
 
Run with small counts for a live demo (see README "Demo Run"); the same
code path scales to the assignment's 1,000+/500,000+ targets by raising
the count flags and, at production scale, sharding the pagination loops
across worker processes (see architecture.md).
"""
from __future__ import annotations
 
import argparse
import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
 
from src.llm.orchestrator import LLMOrchestrator, ExtractionResult
from src.llm.providers import build_default_chain
from src.resolver.entity_resolver import EntityResolver, load_seed_startups
from src.resolver.search_adapters import build_default_web_verifier
from src.schemas.models import (
    StartupRecord, StartupContent, StartupData,
    ProductRecord, ProductContent, PricingModel,
    ResearchPaperRecord, ResearchPaperContent,
    JobRecord, JobContent,
    NewsRecord, NewsContent,
    Source, EntityMappingLogEntry,
)
from src.scrapers.research_papers import scrape_papers_with_stars
from src.scrapers.startups_products import scrape_startups_and_products
from src.scrapers.news_jobs import scrape_all_fresh, NEWS_SOURCES, JOB_SOURCES
from src.utils.dates import normalize_date, SeenStore
from src.utils.http import HttpClient
from src.utils.logging_setup import setup_logging
from src.utils.writers import write_jsonl, write_csv_tab, write_rejects
 
logger = logging.getLogger("graphone.pipeline")
 
 
async def run_pipeline(
    startup_target: int = 50,
    product_target: int = 50,
    paper_target: int = 50,
    include_news: bool = True,
    include_jobs: bool = True,
    output_dir: str = "data/output",
    github_token: str | None = None,
    web_verify_entities: bool = False,
) -> dict:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    setup_logging(run_id=run_id)
    t0 = time.time()
 
    # Web-verification fallback for ambiguous entity names (see
    # src/resolver/entity_resolver.py's resolve_async / src/resolver/web_verifier.py).
    # Opt-in and self-degrading: if the caller didn't ask for it, or asked
    # but no search API key is configured, `verifier` is None and every
    # `resolve_async()` call below behaves exactly like the plain `resolve()`
    # used before this integration -- existing runs are unaffected unless
    # --web-verify-entities is passed AND BRAVE_SEARCH_API_KEY is set.
    verifier = build_default_web_verifier() if web_verify_entities else None
    if web_verify_entities and verifier is None:
        logger.warning(
            "web_verify_entities=True but BRAVE_SEARCH_API_KEY is not set -- "
            "continuing with deterministic-only entity resolution"
        )
    resolver = EntityResolver(canonical_seed=load_seed_startups(), web_verifier=verifier)
    stats: dict = {"run_id": run_id, "phases": {}, "web_verification_enabled": verifier is not None}
 
    async with HttpClient() as client:
 
        # ---- Phase I: Startups & Products ----
        logger.info("Phase I: scraping startups & products (target=%s/%s)", startup_target, product_target)
        t = time.time()
        raw_startups, raw_products = await scrape_startups_and_products(
            client, target_count=max(startup_target, product_target)
        )
        startup_records, product_records, rejects = [], [], []
 
        for s in raw_startups[:startup_target]:
            canonical = await resolver.resolve_async(s["entityName"], entity_kind="STARTUP")
            try:
                rec = StartupRecord(
                    source=Source(name=s["source_name"], url=s["source_url"]),
                    content=StartupContent(
                        entityName=s["entityName"],
                        canonicalName=canonical,
                        data=StartupData(employeeCount=s.get("employeeCount"), location=s.get("location")),
                    ),
                )
                startup_records.append(rec)
            except Exception as e:
                rejects.append({"stage": "startup_validation", "raw": s, "error": str(e)})
 
        for p in raw_products[:product_target]:
            canonical = await resolver.resolve_async(p["startupName"], entity_kind="PRODUCT")
            try:
                rec = ProductRecord(
                    source=Source(name=p["source_name"], url=p["source_url"]),
                    content=ProductContent(
                        productName=p["productName"],
                        startupName=p["startupName"],
                        canonicalStartupName=canonical,
                        pricingModel=PricingModel.UNKNOWN,
                        description=p.get("description"),
                    ),
                )
                product_records.append(rec)
            except Exception as e:
                rejects.append({"stage": "product_validation", "raw": p, "error": str(e)})
 
        stats["phases"]["startups_products"] = {
            "duration_s": round(time.time() - t, 2),
            "startups": len(startup_records), "products": len(product_records),
        }
        logger.info("Phase I done: %s startups, %s products", len(startup_records), len(product_records))
 
        # ---- Phase I: Research papers ----
        logger.info("Phase I: scraping research papers (target=%s)", paper_target)
        t = time.time()
        raw_papers = await scrape_papers_with_stars(client, target_count=paper_target, github_token=github_token)
        paper_records = []
        for pdata in raw_papers:
            dt, _method = normalize_date(pdata.get("published_date"))
            try:
                rec = ResearchPaperRecord(
                    source=Source(name=pdata["source_name"], url=pdata["paper_url"]),
                    content=ResearchPaperContent(
                        title=pdata["title"],
                        authors=pdata.get("authors", []),
                        paper_url=pdata["paper_url"],
                        github_url=pdata.get("github_url"),
                        github_stars=pdata.get("github_stars"),
                        published_date=dt,
                        abstract=pdata.get("abstract"),
                    ),
                )
                paper_records.append(rec)
            except Exception as e:
                rejects.append({"stage": "paper_validation", "raw": pdata, "error": str(e)})
 
        stats["phases"]["research_papers"] = {
            "duration_s": round(time.time() - t, 2), "papers": len(paper_records),
            "with_github": sum(1 for p in paper_records if p.content.github_url),
        }
        logger.info("Phase I done: %s papers (%s with GitHub links)",
                     len(paper_records), stats["phases"]["research_papers"]["with_github"])
 
        # ---- Phase II: News & Jobs (24h freshness) ----
        news_records, job_records = [], []
        seen = SeenStore()
 
        if include_news:
            logger.info("Phase II: scraping news (5 sources, 24h freshness)")
            t = time.time()
            fresh_news = await scrape_all_fresh(client, NEWS_SOURCES, seen_store=seen)
            for n in fresh_news:
                try:
                    rec = NewsRecord(
                        source=Source(name=n["source_name"], url=n["url"]),
                        content=NewsContent(
                            title=n["title"], date=n["published_date"] or datetime.now(timezone.utc),
                            summary=n.get("summary"), url=n["url"],
                        ),
                    )
                    news_records.append(rec)
                except Exception as e:
                    rejects.append({"stage": "news_validation", "raw": n, "error": str(e)})
            stats["phases"]["news"] = {"duration_s": round(time.time() - t, 2), "fresh_records": len(news_records)}
            logger.info("Phase II done: %s fresh news records", len(news_records))
 
        if include_jobs:
            logger.info("Phase II: scraping jobs (5 boards, 24h freshness)")
            t = time.time()
            fresh_jobs = await scrape_all_fresh(client, JOB_SOURCES, seen_store=seen)
            for j in fresh_jobs:
                canonical = await resolver.resolve_async(j.get("source_name", "unknown"), entity_kind="STARTUP")
                try:
                    rec = JobRecord(
                        source=Source(name=j["source_name"], url=j["url"]),
                        content=JobContent(
                            title=j["title"], company=j.get("source_name", "unknown"),
                            canonicalCompany=canonical,
                            date=j["published_date"] or datetime.now(timezone.utc),
                            is_remote=True, url=j["url"],
                        ),
                    )
                    job_records.append(rec)
                except Exception as e:
                    rejects.append({"stage": "job_validation", "raw": j, "error": str(e)})
            stats["phases"]["jobs"] = {"duration_s": round(time.time() - t, 2), "fresh_records": len(job_records)}
            logger.info("Phase II done: %s fresh job records", len(job_records))
 
    # ---- Write outputs ----
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_jsonl(startup_records, f"{output_dir}/startups.jsonl")
    write_jsonl(product_records, f"{output_dir}/products.jsonl")
    write_jsonl(paper_records, f"{output_dir}/research_papers.jsonl")
    write_jsonl(news_records, f"{output_dir}/news.jsonl")
    write_jsonl(job_records, f"{output_dir}/jobs.jsonl")
    write_jsonl(resolver.log, f"{output_dir}/entity_mapping_log.jsonl")
    write_rejects(rejects, f"{output_dir}/rejects.jsonl")
 
    # Web-verification evidence (only populated for entries resolved via
    # web_verified / web_verified_new / web_suggested_low_confidence /
    # web_error -- see EntityResolver.evidence_for). Kept as a separate
    # file rather than a CSV column since most entries have no evidence
    # and it would otherwise pad every row of the Entity Mapping Log tab.
    web_verification_entries = [
        {
            "rawName": entry.rawName,
            "canonicalName": entry.canonicalName,
            "matchMethod": entry.matchMethod,
            "matchScore": entry.matchScore,
            "evidence": resolver.evidence_for(i),
        }
        for i, entry in enumerate(resolver.log)
        if entry.matchMethod.startswith("web_")
    ]
    write_rejects(web_verification_entries, f"{output_dir}/web_verification_log.jsonl")
 
    write_csv_tab(startup_records, f"{output_dir}/csv/Startups.csv")
    write_csv_tab(product_records, f"{output_dir}/csv/Products.csv")
    write_csv_tab(paper_records, f"{output_dir}/csv/Research_Papers.csv")
    write_csv_tab(news_records, f"{output_dir}/csv/News.csv")
    write_csv_tab(job_records, f"{output_dir}/csv/Jobs.csv")
    write_csv_tab(resolver.log, f"{output_dir}/csv/Entity_Mapping_Log.csv")
 
    stats["total_duration_s"] = round(time.time() - t0, 2)
    stats["rejects"] = len(rejects)
    stats["entity_resolutions"] = len(resolver.log)
    stats["web_verification_calls"] = len(web_verification_entries)
 
    import json
    Path(f"{output_dir}/run_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    logger.info("Pipeline run complete in %.2fs. Stats: %s", stats["total_duration_s"], stats)
    return stats
 
 
def main():
    parser = argparse.ArgumentParser(description="GraphOne Intelligence Graph ingestion pipeline")
    parser.add_argument("--startups", type=int, default=50)
    parser.add_argument("--products", type=int, default=50)
    parser.add_argument("--papers", type=int, default=50)
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--no-jobs", action="store_true")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--github-token", default=None)
    parser.add_argument(
        "--web-verify-entities", action="store_true",
        help="Enable web-search + LLM verification for ambiguous entity names "
             "(requires BRAVE_SEARCH_API_KEY; falls back to deterministic-only "
             "resolution with a warning if the key is missing).",
    )
    args = parser.parse_args()
 
    asyncio.run(run_pipeline(
        startup_target=args.startups,
        product_target=args.products,
        paper_target=args.papers,
        include_news=not args.no_news,
        include_jobs=not args.no_jobs,
        output_dir=args.output_dir,
        github_token=args.github_token,
        web_verify_entities=args.web_verify_entities,
    ))
 
 
if __name__ == "__main__":
    main()
