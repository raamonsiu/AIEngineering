"""The pseudonym mapping table.

This is what makes GDPR Article 17 (right to erasure) answerable. Without
it, "remove everything about this person" has no starting point: the
pseudonyms are scattered through the corpus and nothing records that
``Carlos Martinez`` stands for a real individual.

**The store never sees a plaintext value.** It is keyed on an HMAC-SHA256
of the original, computed with a server-side salt that lives in
configuration and never here. That asymmetry is the whole design:

- consistency still works — hash the value, find the row;
- erasure still works — hash the value, delete the row;
- but the table cannot be *enumerated*. A leaked mapping file is a list of
  opaque digests and fake names, not a directory of every real person in
  the corpus.

Storing plaintext would make this file the single highest-value target in
the system: the corpus is pseudonymised, so the only place the real
identities would exist in the clear is here. An HMAC (not a bare hash)
because a plain SHA-256 of a personal name is reversible by dictionary
attack in seconds — the secret salt is what stops that.

``JsonMappingStore`` is the simple backing chosen for now. The
``MappingStore`` protocol is the seam: moving to a database later is a new
implementation of four methods, not a rewrite of the pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable


def hash_value(value: str, salt: str) -> str:
    """HMAC-SHA256 of ``value`` under ``salt``.

    The single place a plaintext personal value is turned into a key. Note
    this is a keyed MAC, not a digest: without the salt the mapping table
    cannot be brute-forced against a name list.
    """
    return hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class PseudonymMapping:
    """One mapping. Note what is absent: the original value."""

    original_hash: str
    pseudonym: str
    entity_type: str
    first_seen_at: str
    # Which catalog source this value first appeared in. Not personal data
    # on its own, and it answers "which sources mention this person?" —
    # step 2 of an erasure request — without walking the whole corpus.
    source_name: str
    occurrences: int = 1


@runtime_checkable
class MappingStore(Protocol):
    """The seam. Four methods, so the backing can change without the
    pipeline noticing."""

    def lookup_or_create(
        self, *, entity_type: str, original_hash: str, factory: Callable[[], str],
        source_name: str,
    ) -> str: ...

    def lookup(self, entity_type: str, original_hash: str) -> Optional[PseudonymMapping]: ...

    def forget(self, original_hash: str) -> list[PseudonymMapping]: ...

    def all_mappings(self) -> list[PseudonymMapping]: ...


class JsonMappingStore:
    """File-backed mapping store.

    Two limitations, stated rather than hidden:

    - **Not encrypted at rest.** The file holds no plaintext values, so a
      leak is far less damaging than it would be otherwise, but the
      pseudonym/hash pairs are still regulated data and in production
      belong behind encryption and access control.
    - **Rewrites the whole file per save.** Fine at this corpus size,
      wrong at a large one.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._by_key: dict[str, PseudonymMapping] = {}
        self._load()

    @staticmethod
    def _key(entity_type: str, original_hash: str) -> str:
        # Keyed on (type, hash), not hash alone: the same string can
        # legitimately be two kinds of entity, and collapsing them would
        # make one erasure silently delete the other.
        return f"{entity_type}\x1f{original_hash}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        self._by_key = {k: PseudonymMapping(**v) for k, v in raw.get("mappings", {}).items()}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mappings": {k: asdict(v) for k, v in self._by_key.items()},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- read ----------------------------------------------------------
    def lookup(self, entity_type: str, original_hash: str) -> Optional[PseudonymMapping]:
        return self._by_key.get(self._key(entity_type, original_hash))

    def find_by_hash(self, original_hash: str) -> list[PseudonymMapping]:
        """Every mapping for a value, across entity types. Erasure starts
        here: a person may appear as a name in one document and as an
        email in another, and one request has to find both."""
        return [m for m in self._by_key.values() if m.original_hash == original_hash]

    def find_by_pseudonym(self, pseudonym: str) -> Optional[PseudonymMapping]:
        return next((m for m in self._by_key.values() if m.pseudonym == pseudonym), None)

    def all_mappings(self) -> list[PseudonymMapping]:
        return list(self._by_key.values())

    # -- write ----------------------------------------------------------
    def lookup_or_create(
        self, *, entity_type: str, original_hash: str, factory: Callable[[], str],
        source_name: str,
    ) -> str:
        """Idempotent: the same ``(entity_type, hash)`` always returns the
        same pseudonym, whatever the call order."""
        key = self._key(entity_type, original_hash)
        existing = self._by_key.get(key)
        if existing is not None:
            self._by_key[key] = PseudonymMapping(
                **{**asdict(existing), "occurrences": existing.occurrences + 1}
            )
            self._flush()
            return existing.pseudonym

        mapping = PseudonymMapping(
            original_hash=original_hash,
            pseudonym=factory(),
            entity_type=entity_type,
            first_seen_at=datetime.now(timezone.utc).isoformat(),
            source_name=source_name,
        )
        self._by_key[key] = mapping
        self._flush()
        return mapping.pseudonym

    def forget(self, original_hash: str) -> list[PseudonymMapping]:
        """Erase every mapping for a hashed value; return what was removed.

        Step 4 of an erasure request. Afterwards the link between the real
        person and any pseudonyms still sitting in an un-reindexed corpus
        is gone, and if they reappear in a future document they get a
        fresh pseudonym unrelated to the old one.

        Steps 2-3 (dropping the affected chunks from the index) and step 5
        (the audit log entry) belong to the caller — which is why the
        removed mappings are returned rather than swallowed.
        """
        removed = self.find_by_hash(original_hash)
        for mapping in removed:
            self._by_key.pop(self._key(mapping.entity_type, mapping.original_hash), None)
        if removed:
            self._flush()
        return removed

