"""The pseudonym mapping table.

This is the piece that makes GDPR Article 17 (right to erasure)
answerable at all. Without it, "remove everything about this person" has
no starting point: the pseudonyms are scattered across the corpus and
nothing records that `Carlos Martinez` stands for a real individual, so
there is nothing to look up and nothing to delete.

Backed by a JSON file. That is the deliberately simple option for now —
the choice of a real encrypted store is revisited when the vector layer
lands, and this interface (``lookup`` / ``save`` / ``forget`` /
``find_by_original``) is the whole surface that would have to be
reimplemented. Two limitations are stated rather than hidden:

- **Not encrypted at rest.** The mapping table is itself personal data
  under GDPR, so in production it needs encryption and its own access
  control. Here it is a plain file kept out of version control.
- **Rewrites the whole file per save.** Fine for a corpus of this size,
  wrong for a large one.

The store is keyed on ``(original_value, entity_type)`` rather than on the
value alone: the same string can legitimately be two different kinds of
entity, and collapsing them would make one erasure delete the other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PseudonymMapping:
    original_value: str
    pseudonym: str
    entity_type: str
    first_seen_at: str
    # Which catalog source this value first appeared in. Persisted so
    # "which sources mention this person?" is answerable without walking
    # the whole corpus — which is step 2 of an erasure request.
    source_name: str
    occurrences: int = 1


class JsonMappingStore:
    """File-backed pseudonym store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._by_key: dict[str, PseudonymMapping] = {}
        self._load()

    @staticmethod
    def _key(original: str, entity_type: str) -> str:
        return f"{entity_type}\x1f{original}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        self._by_key = {
            key: PseudonymMapping(**value) for key, value in raw.get("mappings", {}).items()
        }

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mappings": {key: asdict(value) for key, value in self._by_key.items()},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- read ----------------------------------------------------------
    def lookup(self, original: str, entity_type: str) -> Optional[PseudonymMapping]:
        return self._by_key.get(self._key(original, entity_type))

    def find_by_original(self, original: str) -> list[PseudonymMapping]:
        """Every mapping for a value, across entity types.

        Erasure starts here: a person may appear as a full name in one
        document and as an email in another, and both have to be found
        from one request.
        """
        needle = original.strip().lower()
        return [m for m in self._by_key.values() if m.original_value.strip().lower() == needle]

    def find_by_pseudonym(self, pseudonym: str) -> Optional[PseudonymMapping]:
        return next((m for m in self._by_key.values() if m.pseudonym == pseudonym), None)

    def all_pseudonyms(self) -> list[PseudonymMapping]:
        return list(self._by_key.values())

    # -- write ----------------------------------------------------------
    def save(self, mapping: PseudonymMapping) -> PseudonymMapping:
        self._by_key[self._key(mapping.original_value, mapping.entity_type)] = mapping
        self._flush()
        return mapping

    def touch(self, original: str, entity_type: str) -> None:
        """Count another sighting. The occurrence count is what tells you
        whether a pseudonym appears once or four hundred times, which
        changes how much of the index an erasure request will touch."""
        key = self._key(original, entity_type)
        existing = self._by_key.get(key)
        if existing is not None:
            self._by_key[key] = PseudonymMapping(
                **{**asdict(existing), "occurrences": existing.occurrences + 1}
            )

    def forget(self, original: str) -> list[PseudonymMapping]:
        """Erase every mapping for a value and return what was removed.

        Step 4 of an erasure request. Deleting the mapping is not
        cosmetic: afterwards the link between the real person and the
        pseudonyms left in any un-reindexed corpus is gone, and if the
        person reappears in a future document they receive a fresh
        pseudonym with no relation to the old one.

        The caller is responsible for steps 2-3 (removing the affected
        chunks from the index) and step 5 (the audit log entry). This
        method returns the removed mappings precisely so the caller can
        do both.
        """
        removed = self.find_by_original(original)
        for mapping in removed:
            self._by_key.pop(self._key(mapping.original_value, mapping.entity_type), None)
        if removed:
            self._flush()
        return removed
