import pytest
from pydantic import ValidationError

from src.schemas.models import (
    StartupRecord, StartupContent, Source,
    ResearchPaperRecord, ResearchPaperContent,
    JobRecord, JobContent,
)
from datetime import datetime, timezone


def test_startup_record_round_trips_json():
    rec = StartupRecord(
        source=Source(name="test", url="https://example.com"),
        content=StartupContent(entityName="Acme AI"),
    )
    dumped = rec.model_dump_json()
    assert '"recordType":"STARTUP"' in dumped
    assert rec.schemaVersion == "1.0"


def test_research_paper_rejects_negative_stars():
    with pytest.raises(ValidationError):
        ResearchPaperContent(
            title="Some Paper", paper_url="https://arxiv.org/abs/1234",
            github_stars=-5,
        )


def test_research_paper_allows_missing_github():
    content = ResearchPaperContent(title="Some Paper", paper_url="https://arxiv.org/abs/1234")
    assert content.github_url is None
    assert content.github_stars is None


def test_job_requires_date():
    with pytest.raises(ValidationError):
        JobContent(title="ML Engineer", company="Acme")  # missing required `date`


def test_job_valid():
    j = JobContent(title="ML Engineer", company="Acme", date=datetime.now(timezone.utc))
    assert j.is_remote is False
