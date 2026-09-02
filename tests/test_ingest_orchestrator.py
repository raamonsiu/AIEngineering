"""End-to-end tests for the offline pipeline.

Built on a purpose-made mini-corpus in ``tmp_path`` rather than the
generated one under ``data/``: a test that depends on a corpus someone can
regenerate is a test that fails for reasons unrelated to the code.

The central property here is that the catalog is *obeyed*. Everything else
in the subsystem is replaceable; "a source marked exclude is not indexed,
however present its files are" is the invariant the whole audit rests on.
"""

from __future__ import annotations

import json

import pytest

from app.ingest.anonymization.mapping_store import JsonMappingStore
from app.ingest.anonymization.pseudonymizer import ConsistentPseudonymizer
from app.ingest.audit import generate_audit_report
from app.ingest.catalog import load_catalog
from app.ingest.orchestrator import IngestionPipeline

CATALOG = """
version: 1
last_audited: "2026-09-02"
corpus_root: corpus
sources:
  - name: historical_budgets
    description: Closed budgets.
    location: local://budgets/
    owner_technical: tech@example.com
    owner_business: biz@example.com
    format: json
    pipeline: tabular
    volume: {records: 3, size_mb: 0.1}
    refresh: {declared: monthly, observed_last_update: "2026-08-20", observed_lag_days: 13}
    quality: {completeness: 4, consistency: 3, actuality: 5, reliability: 5}
    sensitivity: {contains_pii: true, pii_types: [client_names], access_restrictions: internal-only}
    lineage: {upstream: erp-finance, transformations: [export_json]}
    decision: include
  - name: meeting_transcripts
    description: Client meetings.
    location: local://transcripts/
    owner_technical: tech@example.com
    owner_business: biz@example.com
    format: txt
    volume: {records: 1, size_mb: 0.1}
    refresh: {declared: weekly, observed_last_update: "2026-08-30", observed_lag_days: 3}
    quality: {completeness: 5, consistency: 2, actuality: 5, reliability: 4}
    sensitivity: {contains_pii: true, pii_types: [personal_names, emails]}
    lineage: {upstream: transcription-service, transformations: [speech_to_text]}
    decision: include
  - name: official_rate_card
    description: Stale rate card.
    location: local://rates/
    owner_technical: finance@example.com
    owner_business: cfo@example.com
    format: xlsx
    volume: {records: 1, size_mb: 0.1}
    refresh: {declared: yearly, observed_last_update: "2024-01-12", observed_lag_days: 963}
    quality: {completeness: 5, consistency: 5, actuality: 1, reliability: 5}
    sensitivity: {contains_pii: false}
    lineage: {upstream: manual-spreadsheet}
    decision: exclude
    notes: Last update January 2024; does not reflect current rates.
"""


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    budgets = root / "budgets"
    transcripts = root / "transcripts"
    rates = root / "rates"
    for folder in (budgets, transcripts, rates):
        folder.mkdir(parents=True)

    def budget(**overrides) -> dict:
        base = {
            "budget_id": "BUDGET-2024-0001", "client_name": "Acme Corp",
            "total_amount": 80000, "currency": "EUR", "hours_estimated": 1280,
            "signed_at": "2024-03-15", "status": "signed",
            "account_manager": "Juan García",
            "phases": [{"name": "Diseño", "hours": 40, "cost_eur": 2500}],
        }
        return {**base, **overrides}

    (budgets / "b1.json").write_text(json.dumps(budget()), encoding="utf-8")
    (budgets / "b2.json").write_text(
        json.dumps(budget(budget_id="BUDGET-2024-0002", client_name="Globex",
                          currency="€", signed_at="15/04/2024")), encoding="utf-8")
    # Contamination: not an ERP identifier. Must be discarded.
    (budgets / "b3.json").write_text(
        json.dumps(budget(budget_id="BDG_BAD", client_name="Initech")), encoding="utf-8")

    (transcripts / "2024_kickoff.txt").write_text(
        "[00:01:12] Ana Ruiz: Necesitamos multi-tenant desde el principio del proyecto.\n"
        "[00:02:03] Luis Soto: Escríbeme a ana.ruiz@acme.com para cerrar el alcance.\n",
        encoding="utf-8",
    )
    # Present on disk, excluded by the catalog. The whole point.
    (rates / "rate_card_2024.xlsx").write_bytes(b"not even a real xlsx")

    catalog_path = tmp_path / "data_catalog.yaml"
    catalog_path.write_text(CATALOG, encoding="utf-8")
    return tmp_path, root, catalog_path


@pytest.fixture
def pipeline(corpus):
    tmp_path, root, catalog_path = corpus
    return IngestionPipeline(
        load_catalog(catalog_path),
        corpus_root=root,
        pseudonymizer=ConsistentPseudonymizer(JsonMappingStore(tmp_path / "pseudonyms.json")),
    )


def test_an_excluded_source_is_not_processed_however_present_its_files(pipeline) -> None:
    run = pipeline.run()
    report = run.report_for("official_rate_card")

    assert report.decision == "exclude"
    assert report.documents_emitted == 0
    assert report.files_seen == 0  # never even listed
    assert "January 2024" in report.skipped_reason
    assert all(d.metadata.source_name != "official_rate_card" for d in run.documents)


def test_contaminated_records_are_discarded_and_counted(pipeline) -> None:
    run = pipeline.run()
    report = run.report_for("historical_budgets")

    assert report.records_valid == 2
    assert report.records_discarded == 1
    assert report.documents_emitted == 2


def test_every_document_carries_catalog_metadata(pipeline) -> None:
    """Traceability by construction: a chunk with no provenance cannot be
    cited, and an un-citable estimate is unusable."""
    run = pipeline.run()

    for document in run.documents:
        assert document.metadata.source_name
        assert document.metadata.source_location
        assert document.metadata.lineage_upstream
        assert document.metadata.ingested_at is not None
        assert document.metadata.document_id


def test_catalog_metadata_overrides_whatever_the_parser_claimed(pipeline) -> None:
    run = pipeline.run()
    budgets = [d for d in run.documents if d.metadata.source_name == "historical_budgets"]

    assert all(d.metadata.lineage_upstream == "erp-finance" for d in budgets)


def test_pii_sources_are_anonymized_and_say_so(pipeline) -> None:
    run = pipeline.run()
    transcripts = [d for d in run.documents if d.metadata.source_name == "meeting_transcripts"]

    assert transcripts
    assert all(d.metadata.anonymized for d in transcripts)
    assert all("anonymize:pseudonymize" in d.metadata.pipeline_stages for d in transcripts)
    blob = "\n".join(d.content for d in transcripts)
    assert "ana.ruiz@acme.com" not in blob
    assert "Ana Ruiz" not in blob


def test_the_speaker_in_metadata_is_anonymized_too(pipeline) -> None:
    """Scrubbing the body and leaving the name in the metadata would leak
    exactly what the body no longer says."""
    run = pipeline.run()
    speakers = {
        d.metadata.document_author
        for d in run.documents
        if d.metadata.source_name == "meeting_transcripts" and d.metadata.document_author
    }

    assert speakers
    assert "Ana Ruiz" not in speakers and "Luis Soto" not in speakers


def test_pipeline_stages_are_recorded_per_document(pipeline) -> None:
    run = pipeline.run()
    budgets = [d for d in run.documents if d.metadata.source_name == "historical_budgets"]

    assert "validate:pandera" in budgets[0].metadata.pipeline_stages
    assert "clean:tabular" in budgets[0].metadata.pipeline_stages


def test_quarantined_records_are_preserved_for_arbitration(corpus, tmp_path) -> None:
    _tmp, root, catalog_path = corpus
    (root / "budgets" / "b4.json").write_text(
        json.dumps({
            "budget_id": "BUDGET-2024-0004", "client_name": "N/A",
            "total_amount": 30000, "currency": "EUR", "hours_estimated": 480,
            "signed_at": "2024-06-01", "status": "signed",
        }),
        encoding="utf-8",
    )
    pipeline = IngestionPipeline(load_catalog(catalog_path), corpus_root=root)

    run = pipeline.run()

    assert run.report_for("historical_budgets").records_quarantined == 1
    assert "historical_budgets" in run.quarantine
    assert "client_name" in run.quarantine["historical_budgets"].iloc[0]["validation_reasons"]


def test_a_source_with_no_pseudonymizer_is_not_marked_anonymized(corpus) -> None:
    """The pair 'declared sensitive but never processed' must stay visible."""
    _tmp, root, catalog_path = corpus
    pipeline = IngestionPipeline(load_catalog(catalog_path), corpus_root=root)

    run = pipeline.run()

    assert all(not d.metadata.anonymized for d in run.documents)
    assert all(d.metadata.contains_pii for d in run.documents)


def test_audit_report_names_the_excluded_source_and_its_reason(pipeline) -> None:
    run = pipeline.run()

    report = generate_audit_report(pipeline.catalog, run)

    assert "official_rate_card" in report
    assert "Fuentes excluidas deliberadamente" in report
    assert "abandonada" in report  # 963 days against a yearly cadence
    assert "no** (actuality)" in report
