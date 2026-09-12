"""
Canonical output schemas for the GraphOne / FrontierAtlas Intelligence Graph.

Every record produced by the pipeline validates against one of these models
before it is written to output. Records that fail validation are routed to
data/rejects.jsonl with the validation error attached, rather than silently
dropped or coerced -- coercing bad data would create exactly the kind of
hallucinated-looking record the assignment explicitly disqualifies for.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RecordType(str, Enum):
    STARTUP = "STARTUP"
    PRODUCT = "PRODUCT"
    RESEARCH_PAPER = "RESEARCH_PAPER"
    JOB = "JOB"
    NEWS = "NEWS"


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"
    UNKNOWN = "UNKNOWN"


class Source(BaseModel):
    name: str
    url: str


class BaseRecord(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: RecordType
    source: Source
    collectedAt: datetime = Field(default_factory=utc_now)

    # Provenance / audit trail required by the "no hallucinated data" rule --
    # every record must trace back to the raw text the LLM was given.
    extractionModel: Optional[str] = None
    rawContentHash: Optional[str] = None


class StartupData(BaseModel):
    employeeCount: Optional[int] = None
    industry: Optional[str] = None
    foundedYear: Optional[int] = None
    location: Optional[str] = None
    fundingStage: Optional[str] = None
    website: Optional[str] = None


class StartupContent(BaseModel):
    entityName: str
    canonicalName: Optional[str] = None  # filled in by the resolver
    data: StartupData = Field(default_factory=StartupData)


class StartupRecord(BaseRecord):
    recordType: RecordType = RecordType.STARTUP
    content: StartupContent


class ProductContent(BaseModel):
    productName: str
    startupName: str
    canonicalStartupName: Optional[str] = None
    pricingModel: PricingModel = PricingModel.UNKNOWN
    description: Optional[str] = None
    category: Optional[str] = None


class ProductRecord(BaseRecord):
    recordType: RecordType = RecordType.PRODUCT
    content: ProductContent


class ResearchPaperContent(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    paper_url: str
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    published_date: Optional[datetime] = None
    abstract: Optional[str] = None

    @field_validator("github_stars")
    @classmethod
    def stars_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("github_stars cannot be negative")
        return v


class ResearchPaperRecord(BaseRecord):
    recordType: RecordType = RecordType.RESEARCH_PAPER
    content: ResearchPaperContent


class JobContent(BaseModel):
    title: str
    company: str
    canonicalCompany: Optional[str] = None
    date: datetime
    is_remote: bool = False
    role_family: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None


class JobRecord(BaseRecord):
    recordType: RecordType = RecordType.JOB
    content: JobContent


class NewsContent(BaseModel):
    title: str
    date: datetime
    full_text: Optional[str] = None
    summary: Optional[str] = None
    url: str
    related_entities: list[str] = Field(default_factory=list)


class NewsRecord(BaseRecord):
    recordType: RecordType = RecordType.NEWS
    content: NewsContent


class EntityMappingLogEntry(BaseModel):
    """One row of the Entity Mapping Log deliverable tab."""
    rawName: str
    canonicalName: str
    entityKind: str  # "STARTUP" | "PRODUCT"
    matchMethod: str  # "exact" | "alias" | "fuzzy" | "unmatched_new"
    matchScore: float
    sourceRecordId: Optional[str] = None
