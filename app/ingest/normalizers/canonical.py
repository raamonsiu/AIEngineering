"""The thin layer that turns a ``ParsedUnit`` into a ``Document``.

Thin on purpose. All it does is merge the three metadata origins — the
catalog's (known before opening the file), the parser's (known only after
reading it) and the pipeline run's (known at processing time) — into the
canonical schema.

Catalog metadata is applied **last and wins**. If a parser set
``source_name`` itself, by bug or by mischief, the catalog's value
overwrites it. The catalog is the authority on which source a document
belongs to, and defence in depth is cheap here.
"""

from __future__ import annotations

from datetime import datetime

from app.ingest.catalog import CatalogSource
from app.ingest.models import Document, DocumentMetadata, ParsedUnit


def to_document(
    unit: ParsedUnit,
    *,
    source: CatalogSource,
    ingested_at: datetime,
    stages: list[str],
    anonymized: bool = False,
) -> Document:
    return Document(
        content=unit.content,
        metadata=DocumentMetadata(
            # --- catalog-provided: authoritative -------------------------
            source_name=source.name,
            source_location=source.location,
            lineage_upstream=source.lineage.upstream,
            contains_pii=source.sensitivity.contains_pii,
            # --- pipeline run --------------------------------------------
            ingested_at=ingested_at,
            pipeline_stages=list(stages),
            anonymized=anonymized,
            # --- parser-provided -----------------------------------------
            document_id=unit.document_id,
            document_title=unit.document_title,
            document_created_at=unit.document_created_at,
            document_author=unit.document_author,
            page_number=unit.page_number,
            section_title=unit.section_title,
            extra=unit.extra,
        ),
    )
