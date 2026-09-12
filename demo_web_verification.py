"""
Demonstrates wiring the web-verification fallback into EntityResolver with
real components: a real web search call and a real LLM adapter from
src/llm/providers.py.
 
The actual adapters live in src/resolver/search_adapters.py (shared with
src/pipeline.py's --web-verify-entities flag) -- this script just shows
how to use them standalone.
 
Requires network access, BRAVE_SEARCH_API_KEY, and at least one of
GEMINI_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY.
 
Run with:  python demo_web_verification.py
"""
import asyncio
 
from src.resolver.entity_resolver import EntityResolver, load_seed_startups
from src.resolver.search_adapters import build_default_web_verifier
 
 
async def main():
    verifier = build_default_web_verifier()
    if verifier is None:
        print("BRAVE_SEARCH_API_KEY is not set -- set it to run this demo with real search.")
        print("(EntityResolver itself handles this gracefully: no verifier configured means")
        print(" resolve_async() just behaves like the deterministic-only resolve().)")
        return
 
    resolver = EntityResolver(canonical_seed=load_seed_startups(), web_verifier=verifier)
 
    # A deliberately ambiguous / rebrand-style test case: not in the seed
    # list verbatim, and not a close-enough spelling match to auto-merge,
    # so this will actually exercise the web verification path.
    test_names = [
        "OpenAI",              # exact match -- verifier NOT called
        "Anthropc",            # typo, high fuzzy score -- verifier NOT called
        "Bard AI",             # ambiguous/rebrand case -- verifier IS called
    ]
 
    for name in test_names:
        canonical = await resolver.resolve_async(name)
        print(f"{name!r:20} -> {canonical!r}")
 
    print("\nFull resolution log:")
    for i, entry in enumerate(resolver.log):
        evidence = resolver.evidence_for(i)
        line = f"  {entry.rawName!r:20} -> {entry.canonicalName!r:20} ({entry.matchMethod}, score={entry.matchScore:.1f})"
        if evidence:
            line += f"\n    evidence: {evidence}"
        print(line)
 
 
if __name__ == "__main__":
    asyncio.run(main())