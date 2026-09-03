"""The thin layer that turns a parser's ``ParsedUnit`` into a ``Document``.

Thin on purpose: it merges the three metadata origins (catalog, parser,
pipeline run) and composes the deterministic document id. Keeping it
separate from the parsers is what lets a parser be tested without
inventing catalog metadata.
"""

from app.ingest.normalizers.canonical import build_document_id, to_document

__all__ = ["build_document_id", "to_document"]
