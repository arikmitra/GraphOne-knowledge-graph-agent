"""
Real provider adapters implementing the LLMProvider interface from
orchestrator.py. Each wraps a vendor's HTTP API behind extract().

These are wired for OpenAI-compatible or native REST calls using aiohttp
directly (no heavy SDK dependency), so the fallback chain works the same
way regardless of vendor. Set the relevant API key env var to activate a
tier; a tier with no key configured simply always raises LLMRetryable-free
LLMBadOutput-free... it raises a clear config error caught by the
orchestrator's fallback loop, so a missing key degrades to "skip to next
tier" rather than crashing the run.
"""
from __future__ import annotations

import os
import aiohttp

from .orchestrator import LLMProvider, LLMRetryable, LLMBadOutput

EXTRACTION_SYSTEM_PROMPT = """You are a precise data extraction engine. Given raw scraped \
text, extract ONLY the fields present in the requested JSON schema. Never invent, \
guess, or hallucinate values. If a field is not present in the text, omit it or set it \
to null. Respond with JSON only, no prose, no markdown fences."""


class GeminiFlashProvider(LLMProvider):
    name = "gemini-flash"
    max_context_tokens = 250_000  # tier 1: large context, cheap, fast

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    async def extract(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise LLMBadOutput("gemini-flash: no API key configured")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    raise LLMRetryable("gemini-flash: 429")
                if resp.status >= 500:
                    raise LLMRetryable(f"gemini-flash: {resp.status}")
                if resp.status != 200:
                    raise LLMBadOutput(f"gemini-flash: HTTP {resp.status}: {await resp.text()}")
                data = await resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    raise LLMBadOutput(f"gemini-flash: unexpected response shape: {e}")


class GroqLlamaProvider(LLMProvider):
    name = "groq-llama3"
    max_context_tokens = 32_000  # tier 2: fast, smaller context window

    def __init__(self, api_key: str | None = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model

    async def extract(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise LLMBadOutput("groq-llama3: no API key configured")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    raise LLMRetryable("groq-llama3: 429")
                if resp.status >= 500:
                    raise LLMRetryable(f"groq-llama3: {resp.status}")
                if resp.status != 200:
                    raise LLMBadOutput(f"groq-llama3: HTTP {resp.status}: {await resp.text()}")
                data = await resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise LLMBadOutput(f"groq-llama3: unexpected response shape: {e}")


class DeepSeekProvider(LLMProvider):
    name = "deepseek"
    max_context_tokens = 64_000  # tier 3: last-resort fallback

    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model

    async def extract(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise LLMBadOutput("deepseek: no API key configured")
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=45)) as resp:
                if resp.status == 429:
                    raise LLMRetryable("deepseek: 429")
                if resp.status >= 500:
                    raise LLMRetryable(f"deepseek: {resp.status}")
                if resp.status != 200:
                    raise LLMBadOutput(f"deepseek: HTTP {resp.status}: {await resp.text()}")
                data = await resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise LLMBadOutput(f"deepseek: unexpected response shape: {e}")


def build_default_chain() -> list[LLMProvider]:
    """
    Standard production fallback chain per the assignment's example:
    Gemini Flash -> Groq Llama 3 -> DeepSeek. Providers with no API key
    set will raise LLMBadOutput immediately, so the orchestrator falls
    through to the next tier without wasting a real retry budget.
    """
    return [GeminiFlashProvider(), GroqLlamaProvider(), DeepSeekProvider()]
