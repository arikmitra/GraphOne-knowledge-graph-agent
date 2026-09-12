"""
Integration tests for the --web-verify-entities wiring in src/pipeline.py.
 
These mock the scraper functions (no live network needed) and the
build_default_web_verifier() factory, to prove:
  1. run_pipeline() with web_verify_entities=False (the default) behaves
     exactly as before -- no verifier constructed, resolve_async() falls
     through to plain deterministic resolution.
  2. web_verify_entities=True with no BRAVE_SEARCH_API_KEY degrades to the
     same behavior, with a warning, rather than crashing.
  3. web_verify_entities=True with a working verifier actually reaches it
     for ambiguous names and the merge shows up in the output files.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch
 
import pytest
 
from src.resolver.entity_resolver import WebVerificationResult
from src.resolver.web_verifier import LLMWebVerifier
 
 
def _mock_scrapers(tmp_path):
    """Patch the three scraper entry points pipeline.py calls, with tiny fixed data."""
    async def fake_scrape_startups_and_products(client, target_count):
        return (
            [{"entityName": "OpenAI, Inc.", "employeeCount": None, "location": None,
              "source_url": "https://example.com/a", "source_name": "test"}],
            [{"productName": "Test Product", "startupName": "OpenAI",
              "description": "x", "source_url": "https://example.com/b", "source_name": "test"}],
        )
 
    async def fake_scrape_papers_with_stars(client, target_count, github_token=None):
        return []
 
    async def fake_scrape_all_fresh(client, sources, seen_store=None):
        return []
 
    return (
        patch("src.pipeline.scrape_startups_and_products", fake_scrape_startups_and_products),
        patch("src.pipeline.scrape_papers_with_stars", fake_scrape_papers_with_stars),
        patch("src.pipeline.scrape_all_fresh", fake_scrape_all_fresh),
    )
 
 
@pytest.mark.asyncio
async def test_default_behavior_unchanged_no_verifier_constructed(tmp_path):
    from src.pipeline import run_pipeline
 
    patches = _mock_scrapers(tmp_path)
    with patches[0], patches[1], patches[2]:
        stats = await run_pipeline(
            startup_target=1, product_target=1, paper_target=0,
            include_news=False, include_jobs=False,
            output_dir=str(tmp_path), web_verify_entities=False,
        )
 
    assert stats["web_verification_enabled"] is False
    assert stats["web_verification_calls"] == 0
    # entity resolution still happened via the deterministic path
    log = (tmp_path / "entity_mapping_log.jsonl").read_text().strip().splitlines()
    assert len(log) >= 1
 
 
@pytest.mark.asyncio
async def test_web_verify_flag_without_api_key_degrades_gracefully(tmp_path, caplog):
    from src.pipeline import run_pipeline
 
    patches = _mock_scrapers(tmp_path)
    env = dict(os.environ)
    env.pop("BRAVE_SEARCH_API_KEY", None)
    with patch.dict(os.environ, env, clear=True), patches[0], patches[1], patches[2]:
        stats = await run_pipeline(
            startup_target=1, product_target=1, paper_target=0,
            include_news=False, include_jobs=False,
            output_dir=str(tmp_path), web_verify_entities=True,
        )
 
    assert stats["web_verification_enabled"] is False  # no key -> disabled, not crashed
    assert stats["web_verification_calls"] == 0
 
 
@pytest.mark.asyncio
async def test_web_verify_enabled_with_working_verifier_reaches_it(tmp_path):
    from src.pipeline import run_pipeline
 
    async def fake_scrape_startups_and_products(client, target_count):
        return (
            [{"entityName": "Bard AI", "employeeCount": None, "location": None,
              "source_url": "https://example.com/a", "source_name": "test"}],
            [],
        )
 
    async def fake_scrape_papers_with_stars(client, target_count, github_token=None):
        return []
 
    async def fake_scrape_all_fresh(client, sources, seen_store=None):
        return []
 
    async def mock_search_fn(query):
        return [{"title": "Bard renamed to Gemini", "snippet": "Google rebranded Bard as Gemini.", "url": "https://x.com"}]
 
    async def mock_llm_fn(system_prompt, user_prompt):
        return json.dumps({
            "is_same_entity": True, "suggested_canonical": "Anthropic",
            "confidence": 0.9, "evidence": "explicit rebrand statement",
        })
 
    fake_verifier = LLMWebVerifier(search_fn=mock_search_fn, llm_complete_fn=mock_llm_fn)
 
    with patch("src.pipeline.scrape_startups_and_products", fake_scrape_startups_and_products), \
         patch("src.pipeline.scrape_papers_with_stars", fake_scrape_papers_with_stars), \
         patch("src.pipeline.scrape_all_fresh", fake_scrape_all_fresh), \
         patch("src.pipeline.build_default_web_verifier", return_value=fake_verifier):
 
        stats = await run_pipeline(
            startup_target=1, product_target=0, paper_target=0,
            include_news=False, include_jobs=False,
            output_dir=str(tmp_path), web_verify_entities=True,
        )
 
    assert stats["web_verification_enabled"] is True
    assert stats["web_verification_calls"] == 1
 
    web_log_path = tmp_path / "web_verification_log.jsonl"
    assert web_log_path.exists()
    entries = [json.loads(line) for line in web_log_path.read_text().strip().splitlines()]
    assert len(entries) == 1
    assert entries[0]["canonicalName"] == "Anthropic"
    assert entries[0]["matchMethod"] == "web_verified"
    assert "rebrand" in entries[0]["evidence"]
 
    # And the startup record itself carries the resolved canonical name.
    startups = [json.loads(line) for line in (tmp_path / "startups.jsonl").read_text().strip().splitlines()]
    assert startups[0]["content"]["canonicalName"] == "Anthropic"