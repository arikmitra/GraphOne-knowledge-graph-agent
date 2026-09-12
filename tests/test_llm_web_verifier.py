import json
 
import pytest
 
from src.resolver.web_verifier import LLMWebVerifier, _parse_json_response
 
 
def make_search_fn(results):
    async def search_fn(query):
        return results
    return search_fn
 
 
def make_llm_fn(response_dict):
    async def llm_fn(system_prompt, user_prompt):
        return json.dumps(response_dict)
    return llm_fn
 
 
@pytest.mark.asyncio
async def test_no_candidates_skips_search_and_llm():
    calls = {"search": 0, "llm": 0}
 
    async def search_fn(q):
        calls["search"] += 1
        return []
 
    async def llm_fn(s, u):
        calls["llm"] += 1
        return "{}"
 
    verifier = LLMWebVerifier(search_fn=search_fn, llm_complete_fn=llm_fn)
    result = await verifier("Some Name", [])
    assert result.is_same_entity is None
    assert calls == {"search": 0, "llm": 0}
 
 
@pytest.mark.asyncio
async def test_empty_search_results_skips_llm_call():
    calls = {"llm": 0}
 
    async def llm_fn(s, u):
        calls["llm"] += 1
        return "{}"
 
    verifier = LLMWebVerifier(search_fn=make_search_fn([]), llm_complete_fn=llm_fn)
    result = await verifier("Some Name", ["Candidate A"])
    assert result.is_same_entity is None
    assert calls["llm"] == 0
 
 
@pytest.mark.asyncio
async def test_happy_path_high_confidence_match():
    search_fn = make_search_fn([
        {"title": "Bard renamed to Gemini", "snippet": "Google officially renamed Bard to Gemini in 2024.", "url": "https://example.com"}
    ])
    llm_fn = make_llm_fn({
        "is_same_entity": True,
        "suggested_canonical": "Google DeepMind",
        "confidence": 0.95,
        "evidence": "Explicit rename statement in search result.",
    })
    verifier = LLMWebVerifier(search_fn=search_fn, llm_complete_fn=llm_fn)
    result = await verifier("Bard", ["Google DeepMind", "OpenAI"])
    assert result.is_same_entity is True
    assert result.suggested_canonical == "Google DeepMind"
    assert result.confidence == 0.95
 
 
@pytest.mark.asyncio
async def test_suggested_canonical_not_in_candidates_is_discarded():
    """If the LLM hallucinates a canonical name we didn't offer, discard the suggestion rather than trust it."""
    search_fn = make_search_fn([{"title": "x", "snippet": "y", "url": "z"}])
    llm_fn = make_llm_fn({
        "is_same_entity": True,
        "suggested_canonical": "Something We Never Offered",
        "confidence": 0.9,
        "evidence": "...",
    })
    verifier = LLMWebVerifier(search_fn=search_fn, llm_complete_fn=llm_fn)
    result = await verifier("Some Name", ["OpenAI", "Anthropic"])
    assert result.suggested_canonical is None  # discarded, not trusted
 
 
@pytest.mark.asyncio
async def test_search_failure_returns_inconclusive_not_exception():
    async def failing_search(q):
        raise ConnectionError("search API timeout")
 
    async def llm_fn(s, u):
        return "{}"
 
    verifier = LLMWebVerifier(search_fn=failing_search, llm_complete_fn=llm_fn)
    result = await verifier("Some Name", ["Candidate"])
    assert result.is_same_entity is None
    assert "search failed" in result.evidence
 
 
@pytest.mark.asyncio
async def test_llm_failure_returns_inconclusive_not_exception():
    async def llm_fn(s, u):
        raise RuntimeError("LLM API down")
 
    verifier = LLMWebVerifier(search_fn=make_search_fn([{"title": "a", "snippet": "b", "url": "c"}]), llm_complete_fn=llm_fn)
    result = await verifier("Some Name", ["Candidate"])
    assert result.is_same_entity is None
    assert "LLM judgment failed" in result.evidence
 
 
@pytest.mark.asyncio
async def test_malformed_llm_json_returns_inconclusive():
    async def bad_llm_fn(s, u):
        return "this is not json"
 
    verifier = LLMWebVerifier(search_fn=make_search_fn([{"title": "a", "snippet": "b", "url": "c"}]), llm_complete_fn=bad_llm_fn)
    result = await verifier("Some Name", ["Candidate"])
    assert result.is_same_entity is None
 
 
def test_parse_json_response_strips_markdown_fences():
    raw = '```json\n{"is_same_entity": true, "confidence": 0.8}\n```'
    parsed = _parse_json_response(raw)
    assert parsed["is_same_entity"] is True
    assert parsed["confidence"] == 0.8
 
 
def test_parse_json_response_plain_json():
    raw = '{"is_same_entity": false, "confidence": 0.1}'
    parsed = _parse_json_response(raw)
    assert parsed["is_same_entity"] is False
 
 
@pytest.mark.asyncio
async def test_max_snippets_caps_search_results_fed_to_llm():
    many_results = [{"title": f"r{i}", "snippet": f"snippet {i}", "url": f"u{i}"} for i in range(20)]
    captured_prompt = {}
 
    async def llm_fn(system_prompt, user_prompt):
        captured_prompt["user"] = user_prompt
        return json.dumps({"is_same_entity": None, "confidence": 0.0, "evidence": "x"})
 
    verifier = LLMWebVerifier(search_fn=make_search_fn(many_results), llm_complete_fn=llm_fn, max_snippets=3)
    await verifier("Some Name", ["Candidate"])
    # only the first 3 snippet markers "[1]", "[2]", "[3]" should appear, not "[4]" onward
    assert "[3]" in captured_prompt["user"]
    assert "[4]" not in captured_prompt["user"]