"""
Concrete `search_fn` / `llm_complete_fn` adapters for `LLMWebVerifier`
(see src/resolver/web_verifier.py), extracted here so both the live
pipeline (src/pipeline.py) and demo_web_verification.py share one
implementation instead of duplicating the wiring.
 
Only built, not auto-activated: build_default_web_verifier() returns None
unless a search API key is actually configured, so importing this module
has zero effect on a pipeline run that hasn't opted in.
"""
from __future__ import annotations
 
import logging
import os
from typing import Optional
 
import aiohttp
 
from src.llm.providers import GeminiFlashProvider, GroqLlamaProvider, DeepSeekProvider
from src.resolver.web_verifier import LLMWebVerifier
 
logger = logging.getLogger("graphone.resolver.search_adapters")
 
 
async def brave_search_fn(query: str) -> list[dict]:
    """
    Web search via the Brave Search API (https://brave.com/search/api/).
    Requires BRAVE_SEARCH_API_KEY. Swap for a different provider by writing
    another async function with this same (query) -> list[{"title",
    "snippet", "url"}] shape and passing it to LLMWebVerifier instead.
    """
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")
 
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": query, "count": 5}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                })
            return results
 
 
async def llm_chain_complete_fn(system_prompt: str, user_prompt: str) -> str:
    """
    Reuses the existing LLM fallback chain (src/llm/providers.py) so the
    web verifier doesn't need its own separate LLM wiring/credentials --
    same Gemini Flash -> Groq Llama 3 -> DeepSeek chain the extraction
    orchestrator uses, tried in order until one responds.
    """
    last_error: Optional[Exception] = None
    for provider in [GeminiFlashProvider(), GroqLlamaProvider(), DeepSeekProvider()]:
        try:
            return await provider.extract(system_prompt, user_prompt)
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"no LLM provider available for web-verification judgment: {last_error}")
 
 
def build_default_web_verifier() -> Optional[LLMWebVerifier]:
    """
    Returns a ready-to-use LLMWebVerifier if a search API key is
    configured, else None. Callers should treat None as "web verification
    is not available in this environment" and proceed without it --
    EntityResolver already handles a None verifier by falling back to
    purely deterministic resolution (see resolve_async's docstring).
 
    This function deliberately does NOT check for LLM provider keys itself
    -- llm_chain_complete_fn already degrades gracefully across the three
    providers, and a missing LLM key surfaces as a per-name inconclusive
    result (logged as web_error) rather than disabling verification
    entirely, since a search key might be set even if an LLM key isn't
    (or vice versa) and partial capability is still useful signal.
    """
    if not os.environ.get("BRAVE_SEARCH_API_KEY"):
        logger.info("BRAVE_SEARCH_API_KEY not set; web-verification fallback disabled")
        return None
    return LLMWebVerifier(search_fn=brave_search_fn, llm_complete_fn=llm_chain_complete_fn)
 
