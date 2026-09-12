"""
A concrete `web_verifier` implementation for EntityResolver.resolve_async
(see src/resolver/entity_resolver.py for the interface it fulfills).
 
Design: search alone can't answer "is X the same company as Y" -- it
returns snippets, not verdicts. So this pairs a search call with an LLM
judgment call: search grounds the LLM in real, current information (so it
isn't just guessing from training data, which could be stale for recent
rebrands/acquisitions), and the LLM turns the snippets into a structured
yes/no/unsure + confidence + evidence answer that EntityResolver can act on.
 
This module intentionally does NOT import a concrete search or LLM client
-- it accepts them as constructor arguments so it can be wired to whatever
is available in the calling context (e.g. Claude's web_search tool when
running inside this environment, or a real HTTP-based search API / one of
the LLMProvider adapters from src/llm/providers.py in a standalone script).
"""
from __future__ import annotations
 
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
 
from src.resolver.entity_resolver import WebVerificationResult
 
logger = logging.getLogger("graphone.resolver.web_verify")
 
VERIFICATION_SYSTEM_PROMPT = """You are verifying whether two company/product names refer to \
the same real-world entity, using search result snippets as evidence. You will be given a raw \
name that needs classification and a list of candidate canonical names it might match, plus \
search snippets about the raw name.
 
Respond with JSON only, no prose, matching this exact shape:
{
  "is_same_entity": true | false | null,
  "suggested_canonical": "<one of the candidate names, or null>",
  "confidence": <float 0.0 to 1.0>,
  "evidence": "<one sentence citing what in the snippets supports this>"
}
 
Rules:
- is_same_entity=true only if the snippets give a concrete reason (rebrand, acquisition,
  known alias, official subsidiary relationship) to believe the raw name IS one of the
  candidates under a different name. Mere topical similarity (e.g. "also an AI company") is
  NOT sufficient grounds for true.
- is_same_entity=false if the snippets make clear this is a distinct, unrelated entity.
- is_same_entity=null if the snippets are inconclusive, absent, or too thin to judge either way.
- confidence should be LOW (below 0.5) whenever the evidence is indirect or you are inferring
  rather than reading an explicit statement. Reserve confidence >= 0.85 for cases where a
  snippet explicitly states the relationship (e.g. "X, formerly known as Y", "X was acquired
  by Y and rebranded", "X (now operating as Y)").
- Never invent a relationship not supported by the snippets. If unsure, say so via null/low
  confidence rather than guessing -- an incorrect high-confidence merge silently corrupts a
  production dataset, which is worse than leaving the entity unresolved for human review.
"""
 
# A search function: query string -> list of {"title", "snippet", "url"} dicts.
SearchFn = Callable[[str], Awaitable[list[dict]]]
 
# An LLM completion function: (system_prompt, user_prompt) -> raw text completion.
LLMCompleteFn = Callable[[str, str], Awaitable[str]]
 
 
@dataclass
class LLMWebVerifier:
    """
    Callable class implementing the `WebVerifier` protocol expected by
    `EntityResolver(web_verifier=...)` / `resolve_async`.
 
    search_fn: performs the actual web search (caller-supplied, so this
        module has no hard dependency on a specific search API).
    llm_complete_fn: performs the actual LLM call (caller-supplied for the
        same reason -- see src/llm/providers.py for ready-made adapters, or
        wire this to Claude's own web_search + reasoning when running
        inside an agent harness that already has both).
    max_snippets: how many search results to feed the LLM. Kept small
        deliberately -- this is a disambiguation check, not a research
        task, and a shorter prompt means cheaper/faster calls at the
        volume an entity resolver runs at.
    """
    search_fn: SearchFn
    llm_complete_fn: LLMCompleteFn
    max_snippets: int = 5
 
    async def __call__(self, raw_name: str, candidate_canonicals: list[str]) -> WebVerificationResult:
        if not candidate_canonicals:
            # Nothing to verify against -- still worth a search in case it
            # reveals this is a well-known alias of something not yet in
            # our canonical set, but without candidates there's nothing to
            # merge into, so we return early rather than spending an LLM
            # call with an empty candidate list.
            return WebVerificationResult(
                is_same_entity=None, suggested_canonical=None, confidence=0.0,
                evidence="no canonical candidates to compare against",
            )
 
        query = f'"{raw_name}" company formerly OR rebrand OR acquired OR "also known as"'
        try:
            results = await self.search_fn(query)
        except Exception as e:
            logger.warning("web_verifier search failed for %r: %s", raw_name, e)
            return WebVerificationResult(
                is_same_entity=None, suggested_canonical=None, confidence=0.0,
                evidence=f"search failed: {e}",
            )
 
        if not results:
            return WebVerificationResult(
                is_same_entity=None, suggested_canonical=None, confidence=0.0,
                evidence="no search results returned",
            )
 
        snippets = results[: self.max_snippets]
        snippet_text = "\n\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('snippet', '')}\nSource: {r.get('url', '')}"
            for i, r in enumerate(snippets)
        )
        user_prompt = (
            f"Raw name to classify: {raw_name!r}\n"
            f"Candidate canonical names: {candidate_canonicals}\n\n"
            f"Search snippets:\n{snippet_text}"
        )
 
        try:
            raw_completion = await self.llm_complete_fn(VERIFICATION_SYSTEM_PROMPT, user_prompt)
            parsed = _parse_json_response(raw_completion)
        except Exception as e:
            logger.warning("web_verifier LLM judgment failed for %r: %s", raw_name, e)
            return WebVerificationResult(
                is_same_entity=None, suggested_canonical=None, confidence=0.0,
                evidence=f"LLM judgment failed: {e}",
            )
 
        suggested = parsed.get("suggested_canonical")
        if suggested is not None and suggested not in candidate_canonicals:
            # The model must only choose from the candidates we gave it --
            # if it returns something else, treat the suggestion as invalid
            # rather than silently trusting a name we didn't offer.
            logger.warning(
                "web_verifier suggested_canonical %r not in candidate list for %r; discarding suggestion",
                suggested, raw_name,
            )
            suggested = None
 
        return WebVerificationResult(
            is_same_entity=parsed.get("is_same_entity"),
            suggested_canonical=suggested,
            confidence=float(parsed.get("confidence", 0.0)),
            evidence=str(parsed.get("evidence", ""))[:500],
        )
 
 
def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw.split("\n", 1)[-1]
    return json.loads(raw)