"""Tests for document identity and for the census.

Both corrections came out of comparing against the reference solution:
documents had no stable per-unit identity (so a re-ingestion would double
the index), and the catalog's facts were measured by hand (so nothing
could tell you when it drifted from reality).
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.ingest.census import check_against_catalog, inspect_folder, inspect_root
from app.ingest.catalog import load_catalog
from app.ingest.models import ParsedUnit
from app.ingest.normalizers.canonical import build_document_id
from app.ingest.orchestrator import IngestionPipeline

# Dates are filled from "today" because the fixture writes the files now:
# a hardcoded audit date would make every run report drift it did not have.
CATALOG = """
version: 3
last_audited: "{today}"
corpus_root: corpus
sources:
  - name: historical_budgets
    description: Closed budgets.
    location: file://budgets/
    owner_technical: t@e.com
    owner_business: b@e.com
    format: json
    pipeline: tabular
    volume: {{records: 2, size_mb: 0.001}}
    refresh: {{declared: monthly, observed_last_update: "{today}", observed_lag_days: 0}}
    quality: {{completeness: 4, consistency: 3, actuality: 5, reliability: 5}}
    sensitivity: {{contains_pii: false}}
    lineage: {{upstream: erp}}
    decision: include
  - name: meeting_transcripts
    description: Client meetings.
    location: file://transcripts/
    owner_technical: t@e.com
    owner_business: b@e.com
    format: txt
    pipeline: text
    volume: {{records: 1, size_mb: 0.001}}
    refresh: {{declared: weekly, observed_last_update: "{today}", observed_lag_days: 0}}
    quality: {{completeness: 5, consistency: 3, actuality: 5, reliability: 4}}
    sensitivity: {{contains_pii: false}}
    lineage: {{upstream: transcription}}
    decision: include
"""


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "budgets").mkdir(parents=True)
    (root / "transcripts").mkdir(parents=True)

    def budget(bid: str) -> str:
        return json.dumps({
            "budget_id": bid, "client_name": "Acme Corp", "client_code": "CLI-1001",
            "total_amount": 80000, "currency": "EUR", "hours_estimated": 1280,
            "signed_at": "2024-03-15", "status": "signed",
        })

    (root / "budgets" / "b1.json").write_text(budget("BUDGET-2024-0001"), encoding="utf-8")
    (root / "budgets" / "b2.json").write_text(budget("BUDGET-2024-0002"), encoding="utf-8")
    (root / "transcripts" / "t1.txt").write_text(
        "[00:01:12] Ana Ruiz: Necesitamos multi-tenant desde el principio del proyecto.\n"
        "[00:02:03] Luis Soto: De acuerdo, lo incluimos en el alcance de la fase uno.\n",
        encoding="utf-8",
    )
    path = tmp_path / "catalog.yaml"
    path.write_text(CATALOG.format(today=date.today().isoformat()), encoding="utf-8")
    return root, path


# ----------------------------------------------------------------------
# Document identity
# ----------------------------------------------------------------------
def test_the_id_is_composed_from_source_document_and_unit() -> None:
    unit = ParsedUnit(content="x", document_id="acta.txt", unit_key="turn-0003")

    assert build_document_id("meeting_transcripts", unit) == (
        "meeting_transcripts:acta.txt:turn-0003"
    )


def test_a_unit_without_a_key_falls_back_to_document_granularity() -> None:
    unit = ParsedUnit(content="x", document_id="BUDGET-2024-0001")

    assert build_document_id("historical_budgets", unit) == (
        "historical_budgets:BUDGET-2024-0001"
    )


def test_every_document_in_a_run_has_a_distinct_id(corpus) -> None:
    """Without a unit key, every turn of a transcript would share one id
    and the index could not tell them apart."""
    root, catalog_path = corpus
    run = IngestionPipeline(load_catalog(catalog_path), corpus_root=root).run()

    ids = [d.id for d in run.documents]
    assert len(ids) == len(set(ids))
    assert all(not r.duplicate_ids for r in run.reports)


def test_ids_are_stable_across_re_ingestion(corpus) -> None:
    """The property the downstream index depends on: re-running replaces
    documents instead of accumulating a second copy of the corpus."""
    root, catalog_path = corpus
    catalog = load_catalog(catalog_path)

    first = [d.id for d in IngestionPipeline(catalog, corpus_root=root).run().documents]
    second = [d.id for d in IngestionPipeline(catalog, corpus_root=root).run().documents]

    assert first == second


def test_every_document_records_the_catalog_version_that_made_it(corpus) -> None:
    """Without it, a document is un-interpretable once the catalog moves on."""
    root, catalog_path = corpus
    run = IngestionPipeline(load_catalog(catalog_path), corpus_root=root).run()

    assert {d.metadata.source_version for d in run.documents} == {"3"}


def test_colliding_ids_are_reported_as_a_defect_of_the_run(corpus) -> None:
    """A collision would make the index silently drop one document, so it
    is surfaced now rather than discovered as missing content later."""
    root, catalog_path = corpus

    class AmbiguousParser:
        supported_formats = {"txt"}

        def parse(self, content: bytes, source_hint: str):
            return [
                ParsedUnit(content="Un turno suficientemente largo para sobrevivir.",
                           document_id="acta.txt", unit_key="same"),
                ParsedUnit(content="Otro turno suficientemente largo para sobrevivir.",
                           document_id="acta.txt", unit_key="same"),
            ]

    from app.ingest.parsers import DEFAULT_PARSERS

    pipeline = IngestionPipeline(
        load_catalog(catalog_path), corpus_root=root,
        parsers={**DEFAULT_PARSERS, "txt": AmbiguousParser()},
    )
    run = pipeline.run()

    report = run.report_for("meeting_transcripts")
    assert report.duplicate_ids == ["meeting_transcripts:acta.txt:same"]
    assert any("duplicate document id" in e for e in report.errors)


# ----------------------------------------------------------------------
# Census
# ----------------------------------------------------------------------
def test_the_census_measures_facts_and_scores_nothing(corpus) -> None:
    root, _ = corpus

    facts = inspect_folder(root / "budgets")

    assert facts.file_count == 2
    assert facts.formats_detected == {"json"}
    assert facts.latest_modified is not None
    assert not hasattr(facts, "quality")


def test_the_census_walks_every_subfolder(corpus) -> None:
    root, _ = corpus

    names = {f.folder.name for f in inspect_root(root)}

    assert names == {"budgets", "transcripts"}


def test_a_truthful_catalog_reports_no_drift(corpus) -> None:
    root, catalog_path = corpus

    assert check_against_catalog(load_catalog(catalog_path), root) == []


def test_a_file_added_without_re_auditing_shows_up_as_drift(corpus) -> None:
    """The failure the census exists to catch: the catalog claiming a
    record count that stopped being true."""
    root, catalog_path = corpus
    (root / "budgets" / "b3.json").write_text("{}", encoding="utf-8")

    drifts = check_against_catalog(load_catalog(catalog_path), root)

    assert [(d.source_name, d.field) for d in drifts] == [
        ("historical_budgets", "volume.records")
    ]
    assert drifts[0].declared == "2" and drifts[0].measured == "3"


def test_a_missing_folder_is_drift_not_a_crash(corpus) -> None:
    root, catalog_path = corpus
    for path in (root / "transcripts").iterdir():
        path.unlink()
    (root / "transcripts").rmdir()

    drifts = check_against_catalog(load_catalog(catalog_path), root)

    assert any(d.source_name == "meeting_transcripts" for d in drifts)
