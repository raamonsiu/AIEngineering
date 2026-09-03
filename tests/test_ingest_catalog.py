"""Tests for the data catalog — the artefact that governs the pipeline.

The property under test is not "YAML parses". It is that the catalog's
decisions are *binding* and that its quality rule composes rather than
averages.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingest.catalog import (
    CatalogSource,
    DataCatalog,
    IngestionDecision,
    Quality,
    load_catalog,
)

MINIMAL_YAML = """
version: 1
last_audited: "2026-09-02"
corpus_root: data/corpus
sources:
  - name: good_source
    description: A source that is fit to index.
    location: drive://x/good/
    owner_technical: tech@example.com
    owner_business: biz@example.com
    format: json
    pipeline: tabular
    volume: {records: 10, size_mb: 1.0}
    refresh: {declared: monthly, observed_last_update: "2026-08-20", observed_lag_days: 13}
    quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 5}
    sensitivity: {contains_pii: true, pii_types: [client_names], access_restrictions: internal-only}
    lineage: {upstream: erp, transformations: [export]}
    decision: include
  - name: stale_source
    description: Officially authoritative, in practice abandoned.
    location: drive://x/stale/
    owner_technical: tech@example.com
    owner_business: biz@example.com
    format: xlsx
    volume: {records: 1, size_mb: 0.1}
    refresh: {declared: yearly, observed_last_update: "2024-01-12", observed_lag_days: 963}
    quality: {completeness: 5, consistency: 5, actuality: 1, reliability: 5}
    sensitivity: {contains_pii: false}
    lineage: {upstream: manual-spreadsheet}
    decision: exclude
    decision_reason: Last update January 2024; rates are stale.
    notes: Does not reflect current rates.
  - name: undecided_source
    description: Needs a human look first.
    location: dropbox://x/review/
    owner_technical: tech@example.com
    owner_business: biz@example.com
    format: txt
    volume: {records: 5, size_mb: 0.5}
    refresh: {declared: weekly, observed_last_update: "2026-08-30", observed_lag_days: 3}
    quality: {completeness: 5, consistency: 2, actuality: 5, reliability: 4}
    sensitivity: {contains_pii: true, pii_types: [personal_names]}
    lineage: {upstream: transcription-service}
    decision: review
    decision_reason: Two incompatible transcript formats coexist.
"""


@pytest.fixture
def catalog(tmp_path) -> DataCatalog:
    path = tmp_path / "data_catalog.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    return load_catalog(path)


def test_quality_composes_rather_than_averages() -> None:
    """A source whose data is complete and possibly false is not a 3."""
    complete_but_unreliable = Quality(
        completeness=5, consistency=5, actuality=5, reliability=1
    )

    assert complete_but_unreliable.is_rag_ready is False
    assert complete_but_unreliable.weakest_dimension == "reliability"
    # The mean would have been 4.0, comfortably "good".
    assert sum([5, 5, 5, 1]) / 4 == 4.0


def test_a_source_is_rag_ready_only_when_no_dimension_is_below_acceptable() -> None:
    assert Quality(completeness=3, consistency=3, actuality=3, reliability=3).is_rag_ready
    assert not Quality(completeness=5, consistency=5, actuality=2, reliability=5).is_rag_ready


def test_only_included_sources_are_offered_to_the_pipeline(catalog: DataCatalog) -> None:
    assert [s.name for s in catalog.included_sources()] == ["good_source"]
    assert [s.name for s in catalog.excluded_sources()] == ["stale_source"]
    assert [s.name for s in catalog.sources_under_review()] == ["undecided_source"]


def test_review_is_not_include(catalog: DataCatalog) -> None:
    """'Under review' must behave as 'not yet', never as a soft yes."""
    names = {s.name for s in catalog.included_sources()}

    assert "undecided_source" not in names


def test_quality_scores_outside_the_scale_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Quality(completeness=7, consistency=3, actuality=3, reliability=3)


def test_location_scheme_resolves_to_a_local_path(catalog, tmp_path) -> None:
    """The catalog keeps the organisation's real scheme; the loader maps it."""
    source = catalog.get("good_source")

    assert source.local_path(tmp_path) == tmp_path / "x/good"


def test_unknown_source_name_raises(catalog: DataCatalog) -> None:
    with pytest.raises(KeyError):
        catalog.get("does_not_exist")


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path) -> None:
    """A mistyped field would otherwise sit in the YAML looking like
    configuration while doing nothing, and the field it was meant to be
    would silently keep its default."""
    typo = MINIMAL_YAML.replace("    format: json", "    formt: json\n    format: json")
    path = tmp_path / "typo.yaml"
    path.write_text(typo, encoding="utf-8")

    with pytest.raises(ValidationError, match="formt"):
        load_catalog(path)


def test_a_non_include_decision_without_a_reason_is_rejected(tmp_path) -> None:
    """An exclusion with no recorded justification is indistinguishable
    from an oversight six months later."""
    without = MINIMAL_YAML.replace(
        "    decision_reason: Last update January 2024; rates are stale.\n", ""
    )
    path = tmp_path / "unjustified.yaml"
    path.write_text(without, encoding="utf-8")

    with pytest.raises(ValidationError, match="decision_reason"):
        load_catalog(path)


def test_include_needs_no_reason(tmp_path) -> None:
    """The burden of justification is on leaving data out, not on using it."""
    catalog = load_catalog(_write(tmp_path, MINIMAL_YAML))

    assert catalog.get("good_source").decision_reason is None


def test_duplicate_source_names_are_rejected(tmp_path) -> None:
    """Two sources sharing a name means one silently shadows the other in
    every lookup, and documents from both claim the same provenance."""
    duplicated = MINIMAL_YAML.replace("name: stale_source", "name: good_source")
    path = tmp_path / "dupes.yaml"
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ValidationError, match="duplicate source name"):
        load_catalog(path)


def test_an_unsupported_format_fails_at_load_not_mid_run(tmp_path) -> None:
    """Otherwise the typo surfaces as 'no parser registered' halfway
    through a run, after other sources have already been processed."""
    typo = MINIMAL_YAML.replace("format: json", "format: jsno")
    path = tmp_path / "badformat.yaml"
    path.write_text(typo, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_catalog(path)


def test_source_names_must_be_usable_as_identifiers(tmp_path) -> None:
    """The name is what a caller passes to the indexing endpoint and what
    every document carries as source_name."""
    shouty = MINIMAL_YAML.replace("name: good_source", "name: Good Source")
    path = tmp_path / "shouty.yaml"
    path.write_text(shouty, encoding="utf-8")

    with pytest.raises(ValidationError, match="snake_case"):
        load_catalog(path)


def _write(tmp_path, text: str):
    path = tmp_path / "catalog.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_malformed_catalog_fails_at_load_not_six_stages_later(tmp_path) -> None:
    broken = MINIMAL_YAML.replace("completeness: 4", "completeness: 99")
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_catalog(path)
