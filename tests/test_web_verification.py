import pytest
 
from src.resolver.entity_resolver import EntityResolver, WebVerificationResult, load_seed_startups
 
 
def make_verifier(response: WebVerificationResult, raise_error: Exception | None = None):
    """Build a mock web_verifier that records calls and returns a fixed response."""
    calls = []
 
    async def verifier(raw_name, candidates):
        calls.append((raw_name, candidates))
        if raise_error:
            raise raise_error
        return response
 
    verifier.calls = calls
    return verifier
 
 
@pytest.mark.asyncio
async def test_no_verifier_configured_behaves_like_sync_path():
    r = EntityResolver(canonical_seed=["OpenAI"])
    result = await r.resolve_async("Completely Unrelated Startup")
    assert result == "Completely Unrelated Startup"
    assert r.log[-1].matchMethod == "unmatched_new"
 
 
@pytest.mark.asyncio
async def test_exact_match_never_calls_verifier():
    verifier = make_verifier(WebVerificationResult(True, "OpenAI", 0.99))
    r = EntityResolver(canonical_seed=["OpenAI"], web_verifier=verifier)
    result = await r.resolve_async("OpenAI, Inc.")
    assert result == "OpenAI"
    assert verifier.calls == []  # deterministic exact match short-circuits before the network call
 
 
@pytest.mark.asyncio
async def test_high_confidence_fuzzy_never_calls_verifier():
    verifier = make_verifier(WebVerificationResult(True, "Anthropic", 0.99))
    r = EntityResolver(canonical_seed=["Anthropic"], web_verifier=verifier)
    result = await r.resolve_async("Anthropc")  # typo, high fuzzy score
    assert result == "Anthropic"
    assert verifier.calls == []
 
 
@pytest.mark.asyncio
async def test_ambiguous_name_triggers_verifier_and_confident_yes_merges():
    verifier = make_verifier(WebVerificationResult(
        is_same_entity=True, suggested_canonical="Google DeepMind",
        confidence=0.95, evidence="Bard was renamed to Gemini by Google DeepMind per official blog.",
    ))
    r = EntityResolver(canonical_seed=["Google DeepMind"], web_verifier=verifier)
    result = await r.resolve_async("Bard AI")
    assert result == "Google DeepMind"
    assert len(verifier.calls) == 1
    assert r.log[-1].matchMethod == "web_verified"
    assert r.log[-1].matchScore == pytest.approx(95.0)
 
 
@pytest.mark.asyncio
async def test_confident_no_mints_new_entity_despite_fuzzy_hint():
    """A name that fuzzy-matches something should NOT auto-merge if the verifier is confident it's different."""
    verifier = make_verifier(WebVerificationResult(
        is_same_entity=False, suggested_canonical=None,
        confidence=0.9, evidence="Distinct company, unrelated to the candidate.",
    ))
    r = EntityResolver(canonical_seed=["Cohere"], web_verifier=verifier)
    result = await r.resolve_async("Cohera Labs")  # close enough to hit the ambiguous band
    assert result == "Cohera Labs"  # minted as its own new entity, NOT merged into "Cohere"
    assert r.log[-1].matchMethod == "web_verified_new"
 
 
@pytest.mark.asyncio
async def test_low_confidence_does_not_auto_merge():
    verifier = make_verifier(WebVerificationResult(
        is_same_entity=True, suggested_canonical="Cohere",
        confidence=0.4, evidence="Weak signal, possibly related.",
    ))
    r = EntityResolver(canonical_seed=["Cohere"], web_verifier=verifier)
    result = await r.resolve_async("Cohera Labs")
    # Low confidence must NOT auto-merge even though is_same_entity=True.
    assert result == "Cohera Labs"
    assert r.log[-1].matchMethod == "web_suggested_low_confidence"
    assert "Cohere" in r.log[-1].canonicalName or result == "Cohera Labs"
 
 
@pytest.mark.asyncio
async def test_inconclusive_result_mints_new_and_logs_for_review():
    verifier = make_verifier(WebVerificationResult(
        is_same_entity=None, suggested_canonical=None, confidence=0.0,
        evidence="no search results returned",
    ))
    r = EntityResolver(canonical_seed=["Cohere"], web_verifier=verifier)
    result = await r.resolve_async("Cohera Labs")
    assert result == "Cohera Labs"
    assert r.log[-1].matchMethod == "web_suggested_low_confidence"
 
 
@pytest.mark.asyncio
async def test_verifier_exception_does_not_crash_pipeline():
    verifier = make_verifier(WebVerificationResult(True, "Cohere", 0.9), raise_error=RuntimeError("search API down"))
    r = EntityResolver(canonical_seed=["Cohere"], web_verifier=verifier)
    result = await r.resolve_async("Some New AI Startup")
    assert result == "Some New AI Startup"
    assert r.log[-1].matchMethod == "web_error"
    assert "search API down" in r.evidence_for(len(r.log) - 1)
 
 
@pytest.mark.asyncio
async def test_completely_new_name_with_no_fuzzy_hint_still_checks_verifier():
    """Even with zero fuzzy candidates, a configured verifier should still be consulted."""
    verifier = make_verifier(WebVerificationResult(None, None, 0.0, "no relevant results"))
    r = EntityResolver(canonical_seed=["OpenAI", "Anthropic"], web_verifier=verifier)
    result = await r.resolve_async("Zylathorp Technologies")
    assert result == "Zylathorp Technologies"
    assert len(verifier.calls) == 1
    assert r.log[-1].matchMethod == "web_suggested_low_confidence"
 
 
@pytest.mark.asyncio
async def test_second_mention_of_web_verified_entity_matches_deterministically():
    """Once web-verified and merged, subsequent mentions should hit the fast exact-match path, not re-verify."""
    verifier = make_verifier(WebVerificationResult(True, "Google DeepMind", 0.95, "explicit rebrand statement"))
    r = EntityResolver(canonical_seed=["Google DeepMind"], web_verifier=verifier)
    first = await r.resolve_async("Bard AI")
    second = await r.resolve_async("Bard AI")
    assert first == second == "Google DeepMind"
    assert len(verifier.calls) == 1  # only called once; second lookup is a cache hit
 
 
@pytest.mark.asyncio
async def test_sync_resolve_ignores_configured_verifier():
    """The plain sync `resolve()` must never attempt to await the verifier."""
    verifier = make_verifier(WebVerificationResult(True, "OpenAI", 0.99))
    r = EntityResolver(canonical_seed=["OpenAI"], web_verifier=verifier)
    result = r.resolve("Totally Ambiguous Name")
    assert result == "Totally Ambiguous Name"
    assert verifier.calls == []