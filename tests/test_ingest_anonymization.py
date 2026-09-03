"""Tests for the pseudonymisation layer and the GDPR machinery.

The mapping store gets most of the attention because it is the piece that
makes erasure answerable; the detection tests are kept to the entity types
this corpus actually declares as PII.

The analyzer fixture is module-scoped: loading a spaCy model costs
seconds, and paying that per test would make the file slow enough that
people stop running it.
"""

from __future__ import annotations

import pytest

from app.ingest.anonymization.mapping_store import (
    JsonMappingStore,
    PseudonymMapping,
    hash_value,
)
from app.ingest.anonymization.pseudonymizer import ConsistentPseudonymizer
from app.ingest.anonymization.recognizers import is_blacklisted


@pytest.fixture
def store(tmp_path) -> JsonMappingStore:
    return JsonMappingStore(tmp_path / "pseudonyms.json")


SALT = "test-salt-not-a-real-secret"


def _mapping(original: str, pseudonym: str, entity_type: str, source: str = "t") -> PseudonymMapping:
    """Build a mapping the way the pipeline does: hashed, never plaintext."""
    return PseudonymMapping(
        original_hash=hash_value(original, SALT),
        pseudonym=pseudonym,
        entity_type=entity_type,
        first_seen_at="2026-09-02T00:00:00+00:00",
        source_name=source,
    )


@pytest.fixture(scope="module")
def _shared_pseudonymizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("anon") / "pseudonyms.json"
    return ConsistentPseudonymizer(JsonMappingStore(path), salt=SALT)


# ----------------------------------------------------------------------
# Mapping store
# ----------------------------------------------------------------------
def test_the_store_never_holds_a_plaintext_value(store, tmp_path) -> None:
    """The point of the whole design: a leaked mapping file must not be a
    directory of the real people in the corpus."""
    store.lookup_or_create(
        entity_type="PERSON",
        original_hash=hash_value("Juan García", SALT),
        factory=lambda: "Carlos Martínez",
        source_name="transcripts",
    )

    on_disk = (tmp_path / "pseudonyms.json").read_text(encoding="utf-8")

    assert "Juan García" not in on_disk
    assert "Carlos Martínez" in on_disk  # the fake name is not a secret
    assert hash_value("Juan García", SALT) in on_disk


def test_the_hash_is_keyed_so_it_is_not_brute_forceable(store) -> None:
    """A bare SHA-256 of a personal name falls to a dictionary attack in
    seconds; the secret salt is what prevents that."""
    assert hash_value("Juan García", "salt-a") != hash_value("Juan García", "salt-b")
    assert hash_value("Juan García", SALT) == hash_value("Juan García", SALT)


def test_mapping_survives_a_reopen(store, tmp_path) -> None:
    """Erasure requests arrive months later, in a different process."""
    digest = hash_value("Juan García", SALT)
    store.lookup_or_create(
        entity_type="PERSON", original_hash=digest,
        factory=lambda: "Carlos Martínez", source_name="t",
    )

    reopened = JsonMappingStore(tmp_path / "pseudonyms.json")

    assert reopened.lookup("PERSON", digest).pseudonym == "Carlos Martínez"


def test_the_same_value_as_two_entity_types_is_two_mappings(store) -> None:
    """Collapsing them would make one erasure silently delete the other."""
    digest = hash_value("Sevilla", SALT)
    store.lookup_or_create(entity_type="LOCATION", original_hash=digest,
                           factory=lambda: "Burgos", source_name="s")
    store.lookup_or_create(entity_type="ORGANIZATION", original_hash=digest,
                           factory=lambda: "Hooli SA", source_name="s")

    assert store.lookup("LOCATION", digest).pseudonym == "Burgos"
    assert store.lookup("ORGANIZATION", digest).pseudonym == "Hooli SA"
    assert len(store.find_by_hash(digest)) == 2


def test_lookup_or_create_is_idempotent(store) -> None:
    digest = hash_value("Ana Ruiz", SALT)
    calls = []

    def factory() -> str:
        calls.append(1)
        return f"Lucía Prat {len(calls)}"

    first = store.lookup_or_create(entity_type="PERSON", original_hash=digest,
                                   factory=factory, source_name="t")
    second = store.lookup_or_create(entity_type="PERSON", original_hash=digest,
                                    factory=factory, source_name="t")

    assert first == second
    assert len(calls) == 1  # the generator ran once


def test_forget_removes_every_mapping_and_reports_what_it_removed(store) -> None:
    """Step 4 of an erasure request; the return value feeds steps 2, 3 and 5."""
    target = hash_value("Juan García", SALT)
    other = hash_value("Ana Ruiz", SALT)
    store.lookup_or_create(entity_type="PERSON", original_hash=target,
                           factory=lambda: "Carlos Martínez", source_name="t")
    store.lookup_or_create(entity_type="EMAIL_ADDRESS", original_hash=target,
                           factory=lambda: "x@y.com", source_name="b")
    store.lookup_or_create(entity_type="PERSON", original_hash=other,
                           factory=lambda: "Lucía Prat", source_name="t")

    removed = store.forget(target)

    assert len(removed) == 2
    assert {m.entity_type for m in removed} == {"PERSON", "EMAIL_ADDRESS"}
    assert store.lookup("PERSON", target) is None
    assert store.lookup("PERSON", other) is not None


def test_forgetting_an_absent_value_is_a_no_op(store) -> None:
    assert store.forget(hash_value("Nunca Existió", SALT)) == []


def test_the_store_records_which_source_first_mentioned_a_person(store) -> None:
    """Step 2 of erasure: which sources mention this person, without
    walking the whole corpus."""
    digest = hash_value("Ana Ruiz", SALT)
    store.lookup_or_create(entity_type="PERSON", original_hash=digest,
                           factory=lambda: "Lucía Prat", source_name="meeting_transcripts")

    assert store.find_by_hash(digest)[0].source_name == "meeting_transcripts"


def test_occurrences_count_up_on_repeat_sightings(store) -> None:
    pseudonymizer = ConsistentPseudonymizer(store, salt=SALT)
    for _ in range(3):
        pseudonymizer.get_or_create_pseudonym("Ana Ruiz", "PERSON", "transcripts")

    assert store.lookup("PERSON", hash_value("Ana Ruiz", SALT)).occurrences == 3


# ----------------------------------------------------------------------
# Pseudonymisation behaviour
# ----------------------------------------------------------------------
def test_the_same_person_gets_the_same_pseudonym_everywhere(_shared_pseudonymizer) -> None:
    """Without this the privacy layer reintroduces exactly the vector-space
    fragmentation the cleaning layer exists to remove."""
    first, _ = _shared_pseudonymizer.anonymize(
        "Juan García confirmó el presupuesto por correo.", source_name="t"
    )
    second, _ = _shared_pseudonymizer.anonymize(
        "Más tarde, Juan García volvió a escribir.", source_name="t"
    )

    replacement = first.split(" confirmó")[0]
    assert "Juan García" not in first and "Juan García" not in second
    assert replacement in second


def test_a_pseudonym_is_the_same_kind_of_thing_as_the_original(_shared_pseudonymizer) -> None:
    """An email becomes an email, not <EMAIL>: the corpus keeps its shape."""
    out, report = _shared_pseudonymizer.anonymize(
        "Escríbeme a contacto@acme.com para cerrar el alcance.", source_name="t"
    )

    assert "contacto@acme.com" not in out
    assert "@" in out
    assert report.by_type.get("EMAIL_ADDRESS") == 1


def test_domain_identifiers_are_detected_and_replaced(_shared_pseudonymizer) -> None:
    """Presidio's defaults know nothing about this company's id formats."""
    out, report = _shared_pseudonymizer.anonymize(
        "El presupuesto BUDGET-2024-0315 corresponde al código CLI-1042.", source_name="b"
    )

    assert "BUDGET-2024-0315" not in out and "CLI-1042" not in out
    assert report.by_type.get("BUDGET_ID") == 1
    assert report.by_type.get("CLIENT_CODE") == 1
    # Shape preserved, so a later parser still recognises the format.
    assert "BUDGET-" in out and "CLI-" in out


def test_spanish_phone_numbers_are_caught(_shared_pseudonymizer) -> None:
    out, _report = _shared_pseudonymizer.anonymize(
        "Su teléfono de contacto es +34 612 345 678.", source_name="t"
    )

    assert "612 345 678" not in out


def test_empty_text_is_returned_untouched(_shared_pseudonymizer) -> None:
    out, report = _shared_pseudonymizer.anonymize("   ", source_name="t")

    assert out == "   " and report.entities_replaced == 0


def test_blacklisted_common_nouns_are_not_treated_as_names() -> None:
    """The Spanish NER model tags ordinary vocabulary as PERSON often
    enough that the threshold alone is not sufficient."""
    assert is_blacklisted("Mar") and is_blacklisted("alcance")
    assert not is_blacklisted("Juan García")


def test_replacement_is_right_to_left_so_offsets_stay_valid(_shared_pseudonymizer) -> None:
    """Several entities in one sentence, with pseudonyms of different
    lengths — a left-to-right pass would corrupt every later offset."""
    out, report = _shared_pseudonymizer.anonymize(
        "Juan García y Ana Ruiz revisaron BUDGET-2024-0001 y escribieron a a@b.com.",
        source_name="t",
    )

    assert report.entities_replaced >= 3
    for leaked in ("Juan García", "Ana Ruiz", "BUDGET-2024-0001", "a@b.com"):
        assert leaked not in out
