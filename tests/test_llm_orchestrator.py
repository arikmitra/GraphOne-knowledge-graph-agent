import json

import pytest

from src.llm.orchestrator import (
    LLMOrchestrator, LLMProvider, LLMRetryable, LLMBadOutput,
    chunk_for_budget,
)


class AlwaysFailProvider(LLMProvider):
    def __init__(self, name):
        self.name = name
        self.max_context_tokens = 1000

    async def extract(self, system_prompt, user_prompt):
        raise LLMRetryable(f"{self.name}: simulated 429")


class AlwaysSucceedProvider(LLMProvider):
    def __init__(self, name, payload):
        self.name = name
        self.max_context_tokens = 1000
        self.payload = payload

    async def extract(self, system_prompt, user_prompt):
        return json.dumps(self.payload)


class BadJsonProvider(LLMProvider):
    def __init__(self, name):
        self.name = name
        self.max_context_tokens = 1000

    async def extract(self, system_prompt, user_prompt):
        return "this is not json at all"


def test_chunk_under_budget_returns_unchanged():
    text = "short text"
    assert chunk_for_budget(text, 1000) == text


def test_chunk_over_budget_truncates_and_keeps_head_and_tail():
    text = "HEAD" * 100 + "MIDDLE" * 1000 + "TAIL" * 100
    out = chunk_for_budget(text, 500)
    assert len(out) <= 500 + len("\n...[truncated]...\n")
    assert out.startswith("HEAD")
    assert out.endswith("TAIL" * 25) or "TAIL" in out[-50:]
    assert "...[truncated]..." in out


@pytest.mark.asyncio
async def test_fallback_chain_falls_through_to_working_provider():
    providers = [
        AlwaysFailProvider("tier1-gemini"),
        AlwaysFailProvider("tier2-groq"),
        AlwaysSucceedProvider("tier3-deepseek", {"title": "hello"}),
    ]
    orch = LLMOrchestrator(providers)
    result = await orch.extract("raw text here", "system prompt")
    assert result.success is True
    assert result.provider_used == "tier3-deepseek"
    assert result.data == {"title": "hello"}
    assert result.attempts == ["tier1-gemini", "tier2-groq", "tier3-deepseek"]


@pytest.mark.asyncio
async def test_all_providers_exhausted_returns_failure_not_exception():
    providers = [AlwaysFailProvider("a"), AlwaysFailProvider("b")]
    orch = LLMOrchestrator(providers)
    result = await orch.extract("text", "system")
    assert result.success is False
    assert result.data is None
    assert "exhausted" in result.error


@pytest.mark.asyncio
async def test_bad_json_falls_through_to_next_tier():
    providers = [BadJsonProvider("bad"), AlwaysSucceedProvider("good", {"ok": True})]
    orch = LLMOrchestrator(providers)
    result = await orch.extract("text", "system")
    assert result.success is True
    assert result.provider_used == "good"


@pytest.mark.asyncio
async def test_first_provider_success_does_not_call_second():
    calls = {"count": 0}

    class CountingProvider(LLMProvider):
        name = "counter"
        max_context_tokens = 1000
        async def extract(self, s, u):
            calls["count"] += 1
            return json.dumps({"x": 1})

    class ShouldNotBeCalled(LLMProvider):
        name = "unused"
        max_context_tokens = 1000
        async def extract(self, s, u):
            raise AssertionError("should not be called")

    orch = LLMOrchestrator([CountingProvider(), ShouldNotBeCalled()])
    result = await orch.extract("text", "system")
    assert result.success is True
    assert calls["count"] == 1
