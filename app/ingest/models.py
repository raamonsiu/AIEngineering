"""The contracts the ingest subsystem is built around.

Two objects matter here, and the distance between them is the whole point
of the three-layer design (loaders -> parsers -> normalizers):

- ``ParsedUnit`` is what a parser emits: text plus whatever metadata the
  *format itself* could yield. A PDF knows its page number, a transcript
  knows its speaker, a DOCX knows its section heading. None of them know
  which catalog source they belong to, or when the pipeline ran.
- ``Document`` is the canonical output of the whole subsystem. Everything
  downstream (chunking, embedding, retrieval) consumes only this.

Keeping them apart is what makes the layers testable in isolation. A
parser test asserts on ``ParsedUnit`` and never has to invent catalog
metadata; a normalizer test asserts on ``Document`` and never has to open
a real PDF. Collapsing them into one object would force every parser test
to fabricate a catalog entry just to satisfy the schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ParsedUnit(BaseModel):
    """One addressable piece of a source document, as the parser sees it.

    A "unit" is whatever the format makes naturally addressable: one
    budget record, one transcript turn, one DOCX section, one PDF page.
    Parsers split at that granularity rather than returning whole files,
    because the smallest unit a *citation* can point at is the smallest
    unit the parser can identify — and citations are the reason this
    pipeline preserves metadata at all.
    """

    content: str
    document_id: str
    # Distinguishes this unit from its siblings inside the same document:
    # a turn number, a page, a section index, a sheet name. The parser
    # knows it; nobody downstream can reconstruct it. It is what makes the
    # canonical Document id addressable below document granularity.
    unit_key: Optional[str] = None
    document_title: Optional[str] = None
    document_created_at: Optional[datetime] = None
    document_author: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    # Format-specific metadata that has no slot in the canonical schema:
    # transcript speaker/timestamp, DOCX template fields, JSON budget
    # status. Preserved without being demanded of every parser.
    extra: dict = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Metadata propagated with every document through the pipeline.

    Three origins, merged by the orchestrator:

    - from the catalog: ``source_name``, ``source_location``,
      ``contains_pii``, ``lineage_upstream``. Known before the document
      is opened, identical for every document of that source.
    - from the parser: title, author, page, section, creation date.
      Known only after reading the document, and only when the format
      carries them.
    - from the pipeline run: ``ingested_at``, ``pipeline_stages``.
      Known at processing time, and what makes a bad batch diagnosable
      after the fact.
    """

    # --- catalog-provided (mandatory for every document) ---------------
    source_name: str
    source_location: str
    ingested_at: datetime
    # The catalog version in force when this document was produced. Without
    # it, a document is un-interpretable after the catalog changes: you
    # cannot tell whether it was built under the rules you are reading now
    # or under the ones that applied three months ago.
    source_version: str

    # --- parser-provided ------------------------------------------------
    document_id: str
    document_title: Optional[str] = None
    document_created_at: Optional[datetime] = None
    document_author: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    # --- lineage & sensitivity -------------------------------------------
    lineage_upstream: Optional[str] = None
    contains_pii: bool = False
    # True once the anonymiser has actually run over ``content``. Distinct
    # from ``contains_pii``, which only says the catalog declared this
    # source sensitive: the pair "declared sensitive but never processed"
    # is precisely the bug worth being able to detect downstream.
    anonymized: bool = False

    # --- pipeline bookkeeping --------------------------------------------
    pipeline_stages: list[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


class Document(BaseModel):
    """The canonical output of the ingest subsystem.

    Every parser, whatever its input format, ends up here. Downstream
    chunking, embedding and retrieval operate exclusively on this type.
    """

    # Deterministic and stable across re-ingestions, composed as
    # ``source:document:unit``. This is what lets the downstream index
    # *replace* a document rather than accumulate a second copy of it every
    # time the pipeline runs — without it, re-indexing silently doubles the
    # corpus and retrieval starts returning the same passage twice.
    #
    # Positional rather than content-derived on purpose: a hash of the text
    # would change whenever an edit changed the text, which is exactly when
    # you most want the id to stay put so the old version is overwritten.
    id: str
    content: str
    metadata: DocumentMetadata
