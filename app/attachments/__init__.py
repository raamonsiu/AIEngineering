"""Online attachment handling. Extraction is re-exported from the single
ingest implementation so callers have one import path, not two."""

from app.attachments.extraction import (
    check_length,
    enforce_length_limit,
    format_attachments_block,
)
from app.ingest.parsers.extraction import ExtractionResult, extract_attachment

__all__ = [
    "ExtractionResult",
    "check_length",
    "enforce_length_limit",
    "extract_attachment",
    "format_attachments_block",
]
