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
from typing import Optional

import yaml
from pydantic import BaseModel, Field


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
    records: int
    size_mb: float


class Refresh(BaseModel):
    """Declared vs observed cadence.

    The gap between the two is the single most revealing field in the
    catalog. A source that officially refreshes monthly and was last
    touched seven months ago is not a monthly source; it is an abandoned
    one that nobody has noticed yet.
    """

    declared: str
    observed_last_update: date
    observed_lag_days: int


class Quality(BaseModel):
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

    upstream: str
    transformations: list[str] = Field(default_factory=list)


class CatalogSource(BaseModel):
    name: str
    description: str
    location: str
    owner_technical: str
    owner_business: str
    format: str
    volume: Volume
    refresh: Refresh
    quality: Quality
    sensitivity: Sensitivity
    lineage: Lineage
    decision: IngestionDecision
    notes: Optional[str] = None
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
    version: int
    last_audited: date
    corpus_root: str
    sources: list[CatalogSource]

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
