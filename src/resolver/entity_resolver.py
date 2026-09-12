"""
Deterministic entity resolution (Phase IV), with an optional web-verification
fallback for ambiguous cases (Phase IV extension).
 
Strategy, in order of precedence:
1. Exact match on a normalized key (lowercase, strip legal suffixes/punct).
2. Known-alias lookup (curated alias -> canonical map).
3. Fuzzy match against the canonical seed list (token_sort_ratio via
   RapidFuzz) above a high similarity threshold -> auto-accepted.
4. AMBIGUOUS BAND (fuzzy score between AMBIGUOUS_FLOOR and FUZZY_THRESHOLD,
   or no fuzzy candidate at all): if a `web_verifier` callback is supplied,
   ask it whether this name is a known alias/rebrand/subsidiary of an
   existing canonical entity. The verifier's answer is only ever a
   *suggestion* -- it is auto-accepted solely when the verifier itself
   reports high confidence; anything else is logged for human review and
   the record still proceeds (minted as new) rather than blocking.
5. No match (and no web verification available/confident) -> treated as a
   new canonical entity in its own right, logged as "unmatched_new" so a
   human can review and merge later.
 
Every decision is logged to the Entity Mapping Log (Phase VI deliverable
tab: "Raw vs Canonical names") with the method and score used, so
resolution is auditable rather than a black box. Web-verification decisions
are logged with method "web_verified" or "web_suggested_low_confidence" so
it's always clear which merges were automated string-matching vs. an
external lookup.
"""
from __future__ import annotations
 
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional
 
from rapidfuzz import fuzz, process
 
from src.schemas.models import EntityMappingLogEntry
 
LEGAL_SUFFIXES = re.compile(
    r"\b(inc\.?|incorporated|llc|ltd\.?|limited|corp\.?|corporation|co\.?|gmbh|plc|"
    r"pvt\.?\s*ltd\.?|pbc|sas|sarl|bv|ag|nv|oy|ab|as|spa|srl)\b",
    re.IGNORECASE,
)
PUNCT = re.compile(r"[.,\-_/&]+")
WHITESPACE = re.compile(r"\s+")
 
FUZZY_THRESHOLD = 90.0      # >= this: auto-accept fuzzy match, no verification needed
AMBIGUOUS_FLOOR = 70.0      # [AMBIGUOUS_FLOOR, FUZZY_THRESHOLD): worth a web check
WEB_VERIFY_ACCEPT_CONFIDENCE = 0.85  # verifier confidence needed to auto-merge
 
 
def normalize_key(name: str) -> str:
    """
    Canonicalization key: lowercase, strip legal suffixes/punctuation, and
    collapse ALL whitespace (not just repeats) so spacing/casing variants
    of the same brand -- "OpenAI" / "Open AI" / "OPEN-AI" -- land on the
    same key. This is intentionally more aggressive than a display-name
    normalizer: it's only used as the lookup key, never shown to a user.
    """
    s = name.strip().lower()
    s = LEGAL_SUFFIXES.sub("", s)
    s = PUNCT.sub(" ", s)
    s = WHITESPACE.sub("", s).strip()  # collapse to zero whitespace for matching
    return s
 
 
@dataclass
class WebVerificationResult:
    """
    The answer a web_verifier callback must return.
 
    is_same_entity: whether the verifier believes raw_name refers to the
        same real-world entity as suggested_canonical (True), a genuinely
        different/new entity (False), or it couldn't determine either way
        (None) -- e.g. no clear search results.
    suggested_canonical: the canonical name the verifier believes this
        should map to. Only meaningful when is_same_entity is True.
    confidence: 0.0-1.0. Only scores >= WEB_VERIFY_ACCEPT_CONFIDENCE are
        auto-merged; anything lower is logged for human review instead.
    evidence: short human-readable justification (e.g. the search snippet
        or reasoning) kept in the log for auditability -- never silently
        discarded, since "why did the resolver think these are the same
        company" is exactly what an auditor will ask.
    """
    is_same_entity: Optional[bool]
    suggested_canonical: Optional[str]
    confidence: float
    evidence: str = ""
 
 
# A web_verifier is an async callable: (raw_name, candidate_canonical_names) -> WebVerificationResult
WebVerifier = Callable[[str, list[str]], Awaitable[WebVerificationResult]]
 
 
@dataclass
class EntityResolver:
    """
    canonical_seed: list of known canonical names, e.g. ["OpenAI", "Anthropic", ...]
    aliases: curated raw-string -> canonical map for known tricky cases
             (rebrands, acquisitions, common misspellings).
    web_verifier: optional async callback used ONLY for ambiguous cases
             (see module docstring). If not supplied, resolution falls back
             to the original deterministic-only behavior -- this keeps the
             resolver usable in contexts with no search tool available
             (e.g. plain unit tests) without any code changes.
    """
    canonical_seed: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    web_verifier: Optional[WebVerifier] = None
 
    _by_key: dict[str, str] = field(init=False, default_factory=dict)
    _alias_by_key: dict[str, str] = field(init=False, default_factory=dict)
    _discovered: dict[str, str] = field(init=False, default_factory=dict)  # new canonicals found at runtime
    log: list[EntityMappingLogEntry] = field(init=False, default_factory=list)
 
    def __post_init__(self):
        self._by_key = {normalize_key(c): c for c in self.canonical_seed}
        self._alias_by_key = {normalize_key(k): v for k, v in self.aliases.items()}
 
    def _all_canonical_keys(self) -> dict[str, str]:
        merged = dict(self._by_key)
        merged.update(self._discovered)
        return merged
 
    def resolve(self, raw_name: str, entity_kind: str = "STARTUP", source_record_id: str | None = None) -> str:
        """
        Synchronous resolution path (original behavior, unchanged): exact ->
        alias -> discovered -> fuzzy(>=90) -> mint new. Does NOT invoke
        web_verifier even if one is configured, since verification requires
        awaiting a network call. Use `resolve_async` when you want the web
        fallback for ambiguous cases; this method remains available so
        existing synchronous call sites keep working unmodified.
        """
        if not raw_name or not raw_name.strip():
            return raw_name
 
        key = normalize_key(raw_name)
 
        hit = self._deterministic_lookup(key)
        if hit is not None:
            canonical, method, score = hit
            self._log(raw_name, canonical, entity_kind, method, score, source_record_id)
            return canonical
 
        # No confident deterministic match: mint as new (sync path can't verify).
        canonical = raw_name.strip()
        self._discovered[key] = canonical
        self._log(raw_name, canonical, entity_kind, "unmatched_new", 0.0, source_record_id)
        return canonical
 
    async def resolve_async(
        self, raw_name: str, entity_kind: str = "STARTUP", source_record_id: str | None = None
    ) -> str:
        """
        Resolution path with the web-verification fallback enabled. Same
        precedence as `resolve` for the deterministic tiers; the difference
        is what happens when nothing deterministic matches confidently:
 
        - If a fuzzy candidate exists in the ambiguous band (or no fuzzy
          candidate exists at all) and `web_verifier` is configured, ask it.
        - A confident "yes, same entity" (confidence >= threshold) merges
          into the suggested canonical and is logged as "web_verified".
        - A confident "no, different entity" mints a new canonical and is
          logged as "web_verified_new" -- this is what lets ambiguous-but-
          actually-different names (e.g. two similarly-named small startups)
          avoid an incorrect fuzzy-adjacent merge.
        - Anything low-confidence or inconclusive is minted as new but
          logged as "web_suggested_low_confidence" (or "web_unavailable" /
          "web_error") with whatever evidence was gathered, so a human
          reviewing the Entity Mapping Log can see it was *considered* and
          resolve it manually -- the pipeline never blocks or fails a
          record over an inconclusive entity match.
        """
        if not raw_name or not raw_name.strip():
            return raw_name
 
        key = normalize_key(raw_name)
 
        hit = self._deterministic_lookup(key)
        if hit is not None:
            canonical, method, score = hit
            self._log(raw_name, canonical, entity_kind, method, score, source_record_id)
            return canonical
 
        # Check whether there's an ambiguous (but sub-threshold) fuzzy candidate
        # worth flagging to the verifier as a hint, even though we won't
        # auto-accept it on string similarity alone.
        candidates = self._all_canonical_keys()
        fuzzy_hint_canonical = None
        fuzzy_hint_score = 0.0
        if candidates:
            match = process.extractOne(key, candidates.keys(), scorer=fuzz.token_sort_ratio)
            if match:
                matched_key, score, _ = match
                fuzzy_hint_score = float(score)
                if fuzzy_hint_score >= AMBIGUOUS_FLOOR:
                    fuzzy_hint_canonical = candidates[matched_key]
 
        if self.web_verifier is None:
            # No verifier configured: behave exactly like the sync path.
            canonical = raw_name.strip()
            self._discovered[key] = canonical
            self._log(raw_name, canonical, entity_kind, "unmatched_new", fuzzy_hint_score, source_record_id)
            return canonical
 
        candidate_pool = list(self._all_canonical_keys().values())
        try:
            result = await self.web_verifier(raw_name, candidate_pool)
        except Exception as e:
            # A verifier failure (network error, API error, etc.) must never
            # take down the pipeline -- fall back to minting as new and log
            # the failure so it's visible in the audit trail.
            canonical = raw_name.strip()
            self._discovered[key] = canonical
            self._log(raw_name, canonical, entity_kind, "web_error", 0.0, source_record_id,
                       evidence=f"verifier raised: {e}")
            return canonical
 
        if result.is_same_entity is True and result.confidence >= WEB_VERIFY_ACCEPT_CONFIDENCE and result.suggested_canonical:
            canonical = result.suggested_canonical
            canon_key = normalize_key(canonical)
            self._discovered.setdefault(canon_key, canonical)
            # Also remember the raw name's own key -> canonical, so a second
            # mention of this exact raw string (e.g. "Bard AI" appearing
            # again in a later batch) hits the fast exact-match path next
            # time instead of re-running a full web verification call.
            self._discovered.setdefault(key, canonical)
            self._log(raw_name, canonical, entity_kind, "web_verified",
                       result.confidence * 100, source_record_id, evidence=result.evidence)
            return canonical
 
        if result.is_same_entity is False and result.confidence >= WEB_VERIFY_ACCEPT_CONFIDENCE:
            # Verifier is confident this is a genuinely different entity,
            # even though it fuzzy-matched something -- mint as new rather
            # than risk merging two different companies.
            canonical = raw_name.strip()
            self._discovered[key] = canonical
            self._log(raw_name, canonical, entity_kind, "web_verified_new",
                       result.confidence * 100, source_record_id, evidence=result.evidence)
            return canonical
 
        # Inconclusive or low-confidence: don't auto-merge. Mint as new but
        # flag clearly in the log for human review, carrying whatever
        # evidence/suggestion the verifier produced.
        canonical = raw_name.strip()
        self._discovered[key] = canonical
        evidence = result.evidence or "no clear signal from web verification"
        if result.suggested_canonical:
            evidence = f"possible match to {result.suggested_canonical!r} ({evidence})"
        self._log(raw_name, canonical, entity_kind, "web_suggested_low_confidence",
                   result.confidence * 100, source_record_id, evidence=evidence)
        return canonical
 
    def _deterministic_lookup(self, key: str) -> Optional[tuple[str, str, float]]:
        """Returns (canonical, method, score) for exact/alias/discovered/high-confidence-fuzzy, else None."""
        if key in self._by_key:
            return self._by_key[key], "exact", 100.0
        if key in self._alias_by_key:
            return self._alias_by_key[key], "alias", 100.0
        if key in self._discovered:
            return self._discovered[key], "exact", 100.0
 
        candidates = self._all_canonical_keys()
        if candidates:
            match = process.extractOne(key, candidates.keys(), scorer=fuzz.token_sort_ratio)
            if match and match[1] >= FUZZY_THRESHOLD:
                matched_key, score, _ = match
                return candidates[matched_key], "fuzzy", float(score)
        return None
 
    def _log(self, raw, canonical, kind, method, score, record_id, evidence: str = ""):
        self.log.append(EntityMappingLogEntry(
            rawName=raw, canonicalName=canonical, entityKind=kind,
            matchMethod=method, matchScore=score, sourceRecordId=record_id,
        ))
        if evidence:
            # Evidence isn't part of the deliverable schema (kept lean for
            # the Sheets tab) -- stash it on a parallel dict for anyone who
            # wants to inspect *why* a web-verification call landed where
            # it did, without bloating the CSV export.
            self._evidence[len(self.log) - 1] = evidence
 
    _evidence: dict[int, str] = field(init=False, default_factory=dict)
 
    def evidence_for(self, log_index: int) -> Optional[str]:
        return self._evidence.get(log_index)
 
 
def load_seed_startups() -> list[str]:
    """~50 known AI startups, per the Phase IV mock-database instruction."""
    return [
        "OpenAI", "Anthropic", "Mistral AI", "Cohere", "Stability AI",
        "Hugging Face", "Scale AI", "Perplexity", "Inflection AI", "Adept AI",
        "Character.AI", "Runway", "ElevenLabs", "Together AI", "Groq",
        "Cerebras Systems", "SambaNova Systems", "Databricks", "Weights & Biases",
        "Pinecone", "Weaviate", "LangChain", "LlamaIndex", "Replicate",
        "Hume AI", "Synthesia", "Jasper", "Writer", "Glean",
        "Sierra", "Harvey", "Cursor (Anysphere)", "Vercel", "Modal Labs",
        "Fireworks AI", "Baseten", "Lambda Labs", "CoreWeave", "Voltage Park",
        "Imbue", "Reka AI", "01.AI", "Moonshot AI", "Zhipu AI",
        "DeepSeek", "xAI", "Suno", "Pika Labs", "Luma AI",
        "World Labs", "Physical Intelligence",
    ]