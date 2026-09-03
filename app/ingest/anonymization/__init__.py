"""Pseudonymisation of personal data, before anything reaches an index.

Three pieces:

- ``recognizers`` wires Presidio to Spanish and adds the identifiers this
  domain uses (``BUDGET_ID``, ``CLIENT_CODE``, local phone numbers).
- ``pseudonymizer`` replaces each detected entity with a stable fake value
  of the same kind, so the corpus keeps its shape instead of collapsing
  onto ``<PERSON>``.
- ``mapping_store`` keeps the link, keyed on a salted HMAC rather than the
  plaintext, which is what makes erasure answerable without the table
  becoming a directory of every real person in the corpus.

This runs before indexing, not as a filter on responses: once a value is
in the vector space it is reachable by any query that comes semantically
close, and no permission check sits in front of that.
"""

from app.ingest.anonymization.mapping_store import (
    JsonMappingStore,
    MappingStore,
    PseudonymMapping,
    hash_value,
)
from app.ingest.anonymization.pseudonymizer import (
    AnonymizationReport,
    ConsistentPseudonymizer,
)
from app.ingest.anonymization.recognizers import build_analyzer

__all__ = [
    "AnonymizationReport",
    "ConsistentPseudonymizer",
    "JsonMappingStore",
    "MappingStore",
    "PseudonymMapping",
    "build_analyzer",
    "hash_value",
]
