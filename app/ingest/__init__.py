"""Offline ingestion: raw enterprise sources in, canonical ``Document`` out.

The subsystem is three layers plus two gates, and the order is the design:

    catalog -> loaders -> parsers -> cleaning -> validation
            -> anonymisation -> normalizers -> Document[]

- ``catalog`` is the control surface, not documentation: the orchestrator
  processes the sources it marks ``include`` and nothing else.
- ``loaders`` know how to reach bytes; ``parsers`` know what is inside
  them; ``normalizers`` turn the result into the canonical contract.
- ``cleaning`` sits deliberately between parser and normalizer, so there
  is exactly one place where data invariants are enforced.
- ``anonymization`` runs before anything is handed on for indexing: once
  a value is in a vector, no permission system can hide it again.

Nothing here is on a user's critical path. That is what lets it spend
minutes on spaCy NER and strict validation (see ``app/routers/indexing.py``
for the offline/online split).
"""

from app.ingest.catalog import CatalogSource, DataCatalog, IngestionDecision, load_catalog
from app.ingest.models import Document, DocumentMetadata, ParsedUnit
from app.ingest.orchestrator import IngestionPipeline, IngestionRun, SourceReport

__all__ = [
    "CatalogSource",
    "DataCatalog",
    "Document",
    "DocumentMetadata",
    "IngestionDecision",
    "IngestionPipeline",
    "IngestionRun",
    "ParsedUnit",
    "SourceReport",
    "load_catalog",
]
