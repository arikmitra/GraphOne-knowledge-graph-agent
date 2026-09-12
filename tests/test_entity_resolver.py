from src.resolver.entity_resolver import EntityResolver, normalize_key


def test_normalize_key_strips_legal_suffix():
    # Whitespace is collapsed entirely in the matching key so spacing
    # variants of the same brand land on one key (see the OpenAI/"Open AI"
    # case from the spec, covered explicitly below).
    assert normalize_key("OpenAI, Inc.") == "openai"
    assert normalize_key("Open AI") == "openai"
    assert normalize_key("OpenAI") == "openai"


def test_exact_match():
    r = EntityResolver(canonical_seed=["OpenAI", "Anthropic"])
    assert r.resolve("OpenAI, Inc.") == "OpenAI"
    assert r.log[-1].matchMethod == "exact"


def test_alias_match():
    r = EntityResolver(canonical_seed=["Meta"], aliases={"Facebook": "Meta"})
    assert r.resolve("Facebook") == "Meta"
    assert r.log[-1].matchMethod == "alias"


def test_fuzzy_match_typo():
    r = EntityResolver(canonical_seed=["Anthropic"])
    result = r.resolve("Anthropc")  # missing an 'i'
    assert result == "Anthropic"
    assert r.log[-1].matchMethod == "fuzzy"
    assert r.log[-1].matchScore >= 90.0


def test_new_entity_becomes_canonical_and_is_reused():
    r = EntityResolver(canonical_seed=["OpenAI"])
    first = r.resolve("Totally New Startup Co.")
    assert first == "Totally New Startup Co."
    assert r.log[-1].matchMethod == "unmatched_new"

    # Second mention of the same (or near-same) raw name should now match
    # the just-discovered canonical rather than minting a duplicate.
    second = r.resolve("Totally New Startup Co")
    assert second == first


def test_the_openai_example_from_spec():
    """OpenAI / OpenAI, Inc. / Open AI must all resolve to canonical 'OpenAI'."""
    r = EntityResolver(canonical_seed=["OpenAI"])
    variants = ["OpenAI", "OpenAI, Inc.", "Open AI"]
    resolved = [r.resolve(v) for v in variants]
    assert all(x == "OpenAI" for x in resolved)


def test_international_legal_suffixes():
    """Non-US legal suffixes (PBC, SAS, GmbH, ...) must also canonicalize."""
    r = EntityResolver(canonical_seed=["Anthropic", "Mistral AI"])
    assert r.resolve("Anthropic PBC") == "Anthropic"
    assert r.resolve("Mistral AI SAS") == "Mistral AI"
