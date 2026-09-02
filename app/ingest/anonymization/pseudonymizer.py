"""Consistent, reversible pseudonymisation.

The choice between the two GDPR-recognised techniques is a retrieval
decision as much as a legal one.

Irreversible anonymisation replaces every detected entity with a generic
token (``<PERSON>``). It is simpler and needs no mapping table — and it
degrades the corpus badly: every person collapses onto the same token, so
two documents about different clients become more similar than two
documents about the same one. It also makes erasure requests
unanswerable, because nothing records who was ever in the corpus.

Reversible pseudonymisation replaces each entity with a *stable fake
value of the same kind*: a name becomes a name, an email becomes an
email. The corpus keeps its shape, and the mapping table makes erasure a
lookup. The cost is that the mapping table is itself personal data and
has to be protected accordingly.

The consistency guarantee is per original value, not per document: the
same "Juan Garcia" becomes the same "Carlos Martinez" in all four hundred
chunks that mention him. Without that, the two chunks land in different
regions of the vector space and retrieval breaks for exactly the reason
inconsistent client spellings break it — the problem the cleaning layer
exists to solve would be reintroduced by the privacy layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from faker import Faker

from app.ingest.anonymization.mapping_store import JsonMappingStore, PseudonymMapping
from app.ingest.anonymization.recognizers import (
    DEFAULT_SCORE_THRESHOLD,
    TARGET_ENTITIES,
    build_analyzer,
    is_blacklisted,
)

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationReport:
    """What the anonymiser did to one piece of text."""

    entities_detected: int
    entities_replaced: int
    by_type: dict[str, int]
    skipped_blacklisted: int = 0


class ConsistentPseudonymizer:
    """Replaces detected entities with stable fake values of the same type."""

    def __init__(
        self,
        store: JsonMappingStore,
        *,
        locale: str = "es_ES",
        seed: int = 20260902,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.store = store
        self.faker = Faker(locale)
        # Seeded so a rebuild of the corpus from scratch produces the same
        # pseudonyms. Unseeded, re-indexing would rename every person and
        # any downstream artefact referring to a pseudonym would rot.
        self.faker.seed_instance(seed)
        self.score_threshold = score_threshold
        self._analyzer = None
        self.generators = {
            "PERSON": self.faker.name,
            "EMAIL_ADDRESS": self.faker.email,
            "PHONE_NUMBER": self.faker.phone_number,
            "LOCATION": self.faker.city,
            "ORGANIZATION": self.faker.company,
            "IBAN_CODE": self.faker.iban,
            "CREDIT_CARD": self.faker.credit_card_number,
            "BUDGET_ID": lambda: f"BUDGET-{self.faker.random_int(2019, 2026)}-"
                                 f"{self.faker.random_int(1000, 9999)}",
            "CLIENT_CODE": lambda: f"CLI-{self.faker.random_int(1000, 9999)}",
        }

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = build_analyzer()
        return self._analyzer

    def get_or_create_pseudonym(self, original: str, entity_type: str, source_name: str) -> str:
        existing = self.store.lookup(original, entity_type)
        if existing is not None:
            self.store.touch(original, entity_type)
            return existing.pseudonym

        generator = self.generators.get(entity_type, self.faker.word)
        pseudonym = generator()
        self.store.save(
            PseudonymMapping(
                original_value=original,
                pseudonym=pseudonym,
                entity_type=entity_type,
                first_seen_at=datetime.now(timezone.utc).isoformat(),
                source_name=source_name,
            )
        )
        return pseudonym

    def anonymize(self, text: str, *, source_name: str) -> tuple[str, AnonymizationReport]:
        """Replace every detected entity, returning the new text and a report."""
        if not text.strip():
            return text, AnonymizationReport(0, 0, {})

        results = self.analyzer.analyze(
            text=text, language="es", entities=list(TARGET_ENTITIES)
        )
        results = [r for r in results if r.score >= self.score_threshold]

        # Longest-match-wins over overlaps, then right-to-left replacement.
        # Replacing left to right would invalidate every later offset the
        # moment a pseudonym differs in length from the original.
        results = _resolve_overlaps(results)
        results.sort(key=lambda r: r.start, reverse=True)

        by_type: dict[str, int] = {}
        replaced = skipped = 0
        out = text
        for result in results:
            original = text[result.start : result.end]
            if is_blacklisted(original):
                skipped += 1
                continue
            pseudonym = self.get_or_create_pseudonym(original, result.entity_type, source_name)
            out = out[: result.start] + pseudonym + out[result.end :]
            by_type[result.entity_type] = by_type.get(result.entity_type, 0) + 1
            replaced += 1

        return out, AnonymizationReport(
            entities_detected=len(results),
            entities_replaced=replaced,
            by_type=by_type,
            skipped_blacklisted=skipped,
        )


def _resolve_overlaps(results: list) -> list:
    """Keep the longest span when detections overlap.

    Recognizers overlap routinely — a PERSON span can sit inside an
    ORGANIZATION one. Replacing both corrupts the text, so the longer
    (more specific) span wins and the shorter is dropped.
    """
    ordered = sorted(results, key=lambda r: (r.start, -(r.end - r.start), -r.score))
    kept: list = []
    for result in ordered:
        if any(result.start < k.end and k.start < result.end for k in kept):
            continue
        kept.append(result)
    return kept
