"""The data catalog: a versioned YAML that the pipeline *obeys*.

This is not documentation that happens to sit next to the code. It is the
control surface: the orchestrator iterates over ``included_sources()``, and
a source marked ``exclude`` is not processed no matter that its files are
sitting right there on disk. Writing the decision down and writing the
decision *into the pipeline* are the same act.

Two reasons it is typed rather than a raw dict. First, a malformed catalog
fails at load time with a field-level error, instead of surfacing six
stages later as a missing metadata key on a chunk. Second, the quality
rubric can carry its own logic (``is_rag_ready``), so "is this source fit
to vectorise?" has exactly one answer in the codebase.

The quality dimensions deliberately do NOT average. A source scoring 5 on
completeness and 1 on reliability is not a 3: it is a source whose data is
complete and possibly false, which is the worst thing you can put in a
retrieval index. Each dimension is a necessary condition, so the rule is
a minimum, not a mean.
"""

from __future__ import annotations

from datetime import date
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IngestionDecision(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    REVIEW = "review"


class QualityScore(IntEnum):
    UNUSABLE = 1
    POOR = 2
    ACCEPTABLE = 3
    GOOD = 4
    EXCELLENT = 5


class Volume(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: int
    size_mb: float


class Refresh(BaseModel):
    """Declared vs observed cadence.

    The gap between the two is the single most revealing field in the
    catalog. A source that officially refreshes monthly and was last
    touched seven months ago is not a monthly source; it is an abandoned
    one that nobody has noticed yet.
    """

    model_config = ConfigDict(extra="forbid")

    declared: str
    observed_last_update: date
    observed_lag_days: int


class Quality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completeness: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    actuality: int = Field(ge=1, le=5)
    reliability: int = Field(ge=1, le=5)

    @property
    def is_rag_ready(self) -> bool:
        """True when no dimension drags below ACCEPTABLE. See module docstring
        for why this is a minimum and not an average."""
        return min(
            self.completeness, self.consistency, self.actuality, self.reliability
        ) >= QualityScore.ACCEPTABLE

    @property
    def weakest_dimension(self) -> str:
        scores = {
            "completeness": self.completeness,
            "consistency": self.consistency,
            "actuality": self.actuality,
            "reliability": self.reliability,
        }
        return min(scores, key=scores.get)  # type: ignore[arg-type]


class Sensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contains_pii: bool
    pii_types: list[str] = Field(default_factory=list)
    access_restrictions: Optional[str] = None


class Lineage(BaseModel):
    """Where the data came from and what has happened to it since.

    Recorded here rather than inferred from filenames because filenames
    are exactly what erodes: `certificacion_ens_2024_marzo.pdf` becomes
    `certificacion_ens.pdf` after two hops between systems, and the
    information that made it interpretable is gone. The catalog is the
    place that context survives by construction.
    """

    model_config = ConfigDict(extra="forbid")

    upstream: str
    transformations: list[str] = Field(default_factory=list)


# Formats the ingest subsystem has a parser for. Declaring this as a
# Literal rather than a free string means a typo ("jsno") fails when the
# catalog loads, not later as "no parser registered for format" halfway
# through a run.
SourceFormat = Literal["json", "txt", "xlsx", "docx", "pdf", "csv"]


class CatalogSource(BaseModel):
    # Unknown keys are rejected rather than ignored. A mistyped field in
    # the YAML would otherwise sit there looking like configuration while
    # doing nothing — and the field it was meant to be would silently keep
    # its default. That is precisely the class of bug a typed catalog
    # exists to prevent.
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    location: str
    owner_technical: str
    owner_business: str
    format: SourceFormat
    volume: Volume
    refresh: Refresh
    quality: Quality
    sensitivity: Sensitivity
    lineage: Lineage
    decision: IngestionDecision
    # Mandatory when the decision is not `include` — see the validator.
    decision_reason: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _name_is_snake_case(cls, value: str) -> str:
        """The name is an identifier: it is what a caller passes to the
        indexing endpoint and what every document carries as
        ``source_name``. Keeping it to lowercase snake_case stops the
        same source being addressable two ways."""
        if not value or not all(c.islower() or c.isdigit() or c == "_" for c in value):
            raise ValueError(f"source name must be lowercase snake_case, got {value!r}")
        return value

    @model_validator(mode="after")
    def _non_include_decisions_need_a_reason(self) -> "CatalogSource":
        """Excluding a source is discipline, not neglect — but only if the
        reason is written down. An exclusion with no recorded justification
        is indistinguishable from an oversight six months later, and
        nobody will dare re-include it or defend it."""
        if self.decision is not IngestionDecision.INCLUDE and not (self.decision_reason or "").strip():
            raise ValueError(
                f"source {self.name!r} has decision={self.decision.value!r} and no "
                f"decision_reason; a non-include decision must be justified in the catalog"
            )
        return self
    # Which pipeline the source takes. "tabular" sources go through the
    # pandas + Pandera cleaning path; "text" sources go through the
    # text-level cleaning path. Declared per source rather than inferred
    # from `format`, because a JSON file can be either depending on what
    # is inside it.
    pipeline: str = "text"

    def local_path(self, corpus_root: Path) -> Path:
        """Resolve ``location`` to a path under ``corpus_root``.

        Locations are written as ``scheme://logical/path`` so the catalog
        stays truthful about where the data really lives in the
        organisation. For this project every scheme resolves to a local
        folder, but keeping the scheme in the YAML means swapping in a
        real Drive or S3 loader later changes the loader, not the catalog.
        """
        _scheme, _, rest = self.location.partition("://")
        return corpus_root / rest.strip("/")


class DataCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    last_audited: date
    corpus_root: str
    sources: list[CatalogSource]

    @field_validator("sources")
    @classmethod
    def _names_are_unique(cls, sources: list[CatalogSource]) -> list[CatalogSource]:
        """Two sources sharing a name means one silently shadows the other
        in every lookup, and documents from both claim the same
        provenance."""
        seen: set[str] = set()
        for source in sources:
            if source.name in seen:
                raise ValueError(f"duplicate source name in catalog: {source.name!r}")
            seen.add(source.name)
        return sources

    def included_sources(self) -> list[CatalogSource]:
        return [s for s in self.sources if s.decision == IngestionDecision.INCLUDE]

    def excluded_sources(self) -> list[CatalogSource]:
        return [s for s in self.sources if s.decision == IngestionDecision.EXCLUDE]

    def sources_under_review(self) -> list[CatalogSource]:
        return [s for s in self.sources if s.decision == IngestionDecision.REVIEW]

    def get(self, name: str) -> CatalogSource:
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"no source named {name!r} in catalog")


def load_catalog(path: Path) -> DataCatalog:
    """Load and validate the data catalog from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return DataCatalog(**raw)
