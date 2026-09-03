"""Tests for the two pipelines exposed as two endpoints.

The property worth protecting is that they stay disjoint: indexing is
accepted and runs in the background, querying is synchronous and (for now)
honestly unimplemented. A service that blurs them is a service that stops
answering questions while it indexes two hundred PDFs.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_data_catalog, get_ingestion_pipeline
from app.ingest.catalog import load_catalog
from app.ingest.orchestrator import IngestionPipeline
from app.main import app

CATALOG = """
version: 1
last_audited: "2026-09-02"
corpus_root: corpus
sources:
  - name: historical_budgets
    description: Closed budgets.
    location: file://budgets/
    owner_technical: t@e.com
    owner_business: b@e.com
    format: json
    pipeline: tabular
    volume: {records: 1, size_mb: 0.1}
    refresh: {declared: monthly, observed_last_update: "2026-08-20", observed_lag_days: 13}
    quality: {completeness: 4, consistency: 3, actuality: 5, reliability: 5}
    sensitivity: {contains_pii: false}
    lineage: {upstream: erp}
    decision: include
  - name: official_rate_card
    description: Stale rates.
    location: file://rates/
    owner_technical: t@e.com
    owner_business: b@e.com
    format: xlsx
    volume: {records: 1, size_mb: 0.1}
    refresh: {declared: yearly, observed_last_update: "2024-01-12", observed_lag_days: 963}
    quality: {completeness: 5, consistency: 5, actuality: 1, reliability: 5}
    sensitivity: {contains_pii: false}
    lineage: {upstream: manual}
    decision: exclude
    decision_reason: Stale since January 2024.
    notes: Stale since January 2024.
"""


@pytest.fixture
def wired(tmp_path):
    root = tmp_path / "corpus"
    (root / "budgets").mkdir(parents=True)
    (root / "rates").mkdir(parents=True)
    (root / "budgets" / "b1.json").write_text(
        json.dumps({
            "budget_id": "BUDGET-2024-0001", "client_name": "Acme Corp",
            "total_amount": 80000, "currency": "EUR", "hours_estimated": 1280,
            "signed_at": "2024-03-15", "status": "signed",
        }),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(CATALOG, encoding="utf-8")
    catalog = load_catalog(catalog_path)

    app.dependency_overrides[get_data_catalog] = lambda: catalog
    app.dependency_overrides[get_ingestion_pipeline] = lambda: IngestionPipeline(
        catalog, corpus_root=root
    )
    yield catalog
    app.dependency_overrides.pop(get_data_catalog, None)
    app.dependency_overrides.pop(get_ingestion_pipeline, None)


# ----------------------------------------------------------------------
# Offline pipeline
# ----------------------------------------------------------------------
def test_indexing_is_accepted_not_awaited(client: TestClient, wired) -> None:
    """202, not 200: the caller gets a receipt, not a result."""
    response = client.post("/api/v1/index/run", json={})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "scheduled"
    assert body["scheduled_sources"] == ["historical_budgets"]


def test_the_caller_cannot_schedule_an_excluded_source_into_the_index(
    client: TestClient, wired
) -> None:
    """Naming a source does not override its catalog decision."""
    run_id = client.post(
        "/api/v1/index/run", json={"sources": ["official_rate_card"]}
    ).json()["run_id"]

    status = client.get(f"/api/v1/index/runs/{run_id}").json()

    assert status["status"] == "completed"
    assert status["document_count"] == 0


def test_an_unknown_source_is_rejected_before_scheduling(client: TestClient, wired) -> None:
    response = client.post("/api/v1/index/run", json={"sources": ["nope"]})

    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_a_completed_run_reports_per_source_counts(client: TestClient, wired) -> None:
    run_id = client.post("/api/v1/index/run", json={}).json()["run_id"]

    status = client.get(f"/api/v1/index/runs/{run_id}").json()

    assert status["status"] == "completed"
    assert status["document_count"] == 1
    by_name = {r["source_name"]: r for r in status["reports"]}
    assert by_name["historical_budgets"]["records_valid"] == 1
    assert by_name["official_rate_card"]["decision"] == "exclude"
    assert by_name["official_rate_card"]["documents_emitted"] == 0


def test_unknown_run_id_is_404(client: TestClient, wired) -> None:
    assert client.get("/api/v1/index/runs/does-not-exist").status_code == 404


def test_audit_report_is_served_as_markdown(client: TestClient, wired) -> None:
    body = client.get("/api/v1/index/audit").json()

    assert body["markdown"].startswith("# Reporte de auditoría de datos")
    assert "official_rate_card" in body["markdown"]


# ----------------------------------------------------------------------
# Online pipeline
# ----------------------------------------------------------------------
def test_query_is_501_not_404(client: TestClient) -> None:
    """404 would say 'no such endpoint', which is false: the contract
    exists and the implementation is pending."""
    response = client.post("/api/v1/query", json={"user_question": "¿Cuánto costó el último CRM?"})

    assert response.status_code == 501
    assert response.json()["detail"]["reason"] == "retrieval_not_implemented"


def test_query_still_validates_its_payload(client: TestClient) -> None:
    """A caller integrating today finds out today whether their body is
    right, instead of discovering it when retrieval lands."""
    response = client.post("/api/v1/query", json={"user_question": "x"})

    assert response.status_code == 422


def test_query_rejects_an_out_of_range_top_k(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query", json={"user_question": "¿Cuánto costó el CRM?", "top_k": 999}
    )

    assert response.status_code == 422
