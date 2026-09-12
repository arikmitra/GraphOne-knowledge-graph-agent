"""
Multi-tier LLM extraction engine (Phase III).

Fallback chain: provider[0] -> provider[1] -> provider[2] -> ... Each
provider is tried in order; on a retryable failure (429, 5xx, malformed
JSON) we fall to the next tier rather than failing the whole record.

Chunking: `chunk_for_budget` truncates/splits raw text so payloads stay
under each provider's context budget while keeping the most
information-dense parts (title/headline first, then the beginning and end
of the body, where bylines/dates/summaries usually live, dropping the
noisy middle first).

This module is provider-agnostic: each tier is a small adapter implementing
`LLMProvider.extract(prompt) -> str`. Swap in real SDK calls (google-genai,
groq, openai-compatible clients, etc.) behind the same interface.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger("graphone.llm")


class LLMRetryable(Exception):
    """Raised by a provider adapter on 429 / transient server error."""


class LLMBadOutput(Exception):
    """Raised when a provider returns output that fails schema validation."""


@dataclass
class ExtractionResult:
    data: Optional[dict]
    provider_used: Optional[str]
    attempts: list[str]
    raw_text_hash: str
    success: bool
    error: Optional[str] = None


class LLMProvider(ABC):
    name: str
    max_context_tokens: int

    @abstractmethod
    async def extract(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion. Raise LLMRetryable on 429/5xx."""


class MockProvider(LLMProvider):
    """
    Deterministic stand-in used for local testing and the demo run so the
    pipeline is fully exercisable without live API keys. Swap for
    GeminiFlashProvider / GroqLlamaProvider / DeepSeekProvider in production
    -- same interface, see providers.py.
    """

    def __init__(self, name: str, max_context_tokens: int, fail_rate: float = 0.0):
        self.name = name
        self.max_context_tokens = max_context_tokens
        self.fail_rate = fail_rate

    async def extract(self, system_prompt: str, user_prompt: str) -> str:
        import random
        await asyncio.sleep(0.01)
        if random.random() < self.fail_rate:
            raise LLMRetryable(f"{self.name}: simulated 429")
        # Extremely small heuristic "extraction" so the demo produces
        # plausible structured output without a live model call.
        return _heuristic_extract(user_prompt)


def _heuristic_extract(text: str) -> str:
    """Toy extractor for offline demo runs — not used when a real provider is wired in."""
    import re
    title_match = re.search(r"(?:^|\n)([A-Z][^\n]{10,120})", text)
    title = title_match.group(1).strip() if title_match else text[:80].strip()
    return json.dumps({"title": title, "confidence": "heuristic"})


def chunk_for_budget(text: str, max_chars: int) -> str:
    """
    Keep the response under the provider's payload budget (Phase III:
    never trigger 413) while retaining the most information-dense spans:
    the first ~70% of the budget from the start (headline, byline, lede/
    abstract) and the last ~30% from the end (often has dates, author
    bios, footer metadata) -- dropping the noisy middle first.
    """
    if len(text) <= max_chars:
        return text
    head_budget = int(max_chars * 0.7)
    tail_budget = max_chars - head_budget - len("\n...[truncated]...\n")
    return text[:head_budget] + "\n...[truncated]...\n" + text[-tail_budget:]


class LLMOrchestrator:
    def __init__(self, providers: list[LLMProvider], schema_validator=None):
        """
        providers: ordered fallback chain, e.g.
            [GeminiFlashProvider(), GroqLlamaProvider(), DeepSeekProvider()]
        schema_validator: optional callable(dict) -> dict that raises on
            invalid shape, e.g. a pydantic model's `.model_validate`.
        """
        self.providers = providers
        self.schema_validator = schema_validator
        self.stats = {p.name: {"attempts": 0, "successes": 0, "failures": 0} for p in providers}

    async def extract(self, raw_text: str, system_prompt: str) -> ExtractionResult:
        import hashlib
        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        attempts: list[str] = []

        for provider in self.providers:
            chunked = chunk_for_budget(raw_text, provider.max_context_tokens * 3)  # ~3 chars/token
            attempts.append(provider.name)
            self.stats[provider.name]["attempts"] += 1
            try:
                raw_out = await self._call_with_retry(provider, system_prompt, chunked)
                parsed = self._parse_json(raw_out)
                if self.schema_validator:
                    parsed = self.schema_validator(parsed)
                self.stats[provider.name]["successes"] += 1
                return ExtractionResult(
                    data=parsed,
                    provider_used=provider.name,
                    attempts=attempts,
                    raw_text_hash=raw_hash,
                    success=True,
                )
            except Exception as e:
                self.stats[provider.name]["failures"] += 1
                logger.warning("provider %s failed, falling back: %s", provider.name, e)
                continue

        return ExtractionResult(
            data=None,
            provider_used=None,
            attempts=attempts,
            raw_text_hash=raw_hash,
            success=False,
            error="all providers in fallback chain exhausted",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=20, jitter=2),
           retry=lambda retry_state: isinstance(retry_state.outcome.exception(), LLMRetryable))
    async def _call_with_retry(self, provider: LLMProvider, system_prompt: str, text: str) -> str:
        return await provider.extract(system_prompt, text)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[-1] if raw.lower().startswith("json") else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMBadOutput(f"non-JSON output: {e}") from e
