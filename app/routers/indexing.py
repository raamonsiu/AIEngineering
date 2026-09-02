"""The two halves of a RAG system, as two endpoints with nothing in common.

``POST /api/v1/index/run`` is the OFFLINE pipeline: ingest, parse, clean,
validate, anonymise. Triggered by ingestion events, never by a user
question. Budget: minutes to hours. Returns immediately with a run id;
the work happens in the background.

``POST /api/v1/query`` is the ONLINE pipeline: retrieve, augment,
generate. Triggered by a user who is waiting. Budget: under three
seconds. It currently returns **501** — retrieval needs a vector index,
and there is none until the corpus has been through this subsystem and
been embedded. The contract is written down now so the shape of the API
is fixed before anything depends on it.

Keeping them apart is the architectural decision, not a stylistic one. A
service that indexes two hundred PDFs inside the same request path that
answers questions is a service that stops answering questions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_data_catalog, get_ingestion_pipeline
from app.ingest.audit import generate_audit_report
from app.ingest.catalog import DataCatalog
from app.ingest.orchestrator import IngestionPipeline, IngestionRun

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["indexing"])

# Process-local run registry, same trade-off as the session store: this
# is operational state for one worker's lifetime, not a system of record.
# With several workers each would see only its own runs.
_RUNS: dict[str, dict] = {}


class IndexRunRequest(BaseModel):
    """Which sources to index. Empty means every source the catalog
    marks ``include`` — the catalog decides, not the caller."""

    sources: list[str] = Field(default_factory=list)


class IndexRunAccepted(BaseModel):
    run_id: str
    status: str
    scheduled_sources: list[str]


class SourceReportOut(BaseModel):
    source_name: str
    decision: str
    files_seen: int
    units_parsed: int
    units_after_cleaning: int
    documents_emitted: int
    records_valid: int
    records_quarantined: int
    records_discarded: int
    entities_anonymized: int
    divergent_duplicates: list[str]
    errors: list[str]
    skipped_reason: str | None


class IndexRunStatus(BaseModel):
    run_id: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    document_count: int = 0
    reports: list[SourceReportOut] = Field(default_factory=list)
    error: str | None = None


def _execute(run_id: str, pipeline: IngestionPipeline, sources: list[str]) -> None:
    """Background worker. Never raises: a failed run is a recorded status,
    not a lost task with a traceback in a log nobody reads."""
    try:
        if sources:
            # Narrowing to named sources still respects each source's own
            # catalog decision — asking for an excluded source does not
            # override the exclusion, it just does nothing.
            pipeline.catalog = pipeline.catalog.model_copy(
                update={"sources": [s for s in pipeline.catalog.sources if s.name in sources]}
            )
        run: IngestionRun = pipeline.run()
        _RUNS[run_id] = {
            "status": "completed",
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "document_count": run.document_count,
            "reports": [SourceReportOut(**vars(r)) for r in run.reports],
            "run": run,
        }
        log.info("index_run_completed", run_id=run_id, documents=run.document_count)
    except Exception as exc:  # noqa: BLE001
        _RUNS[run_id] = {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc),
            "error": f"{type(exc).__name__}: {exc}",
        }
        log.error("index_run_failed", run_id=run_id, error=str(exc)[:400])


@router.post("/index/run", response_model=IndexRunAccepted, status_code=202)
def trigger_indexing(
    request: IndexRunRequest,
    tasks: BackgroundTasks,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    catalog: DataCatalog = Depends(get_data_catalog),
) -> IndexRunAccepted:
    """Schedule an indexing run. Returns 202 without waiting for it."""
    known = {s.name for s in catalog.sources}
    unknown = sorted(set(request.sources) - known)
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"unknown source(s): {unknown}; known: {sorted(known)}"
        )

    scheduled = request.sources or [s.name for s in catalog.included_sources()]
    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {"status": "running", "started_at": datetime.now(timezone.utc)}
    tasks.add_task(_execute, run_id, pipeline, request.sources)

    log.info("index_run_scheduled", run_id=run_id, sources=scheduled)
    return IndexRunAccepted(run_id=run_id, status="scheduled", scheduled_sources=scheduled)


@router.get("/index/runs/{run_id}", response_model=IndexRunStatus)
def get_run_status(run_id: str) -> IndexRunStatus:
    state = _RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown run id.")
    return IndexRunStatus(run_id=run_id, **{k: v for k, v in state.items() if k != "run"})


@router.get("/index/audit")
def get_audit_report(
    run_id: str | None = None, catalog: DataCatalog = Depends(get_data_catalog)
) -> dict:
    """The audit report as Markdown, optionally enriched with a run."""
    run = (_RUNS.get(run_id) or {}).get("run") if run_id else None
    return {"markdown": generate_audit_report(catalog, run)}


class QueryRequest(BaseModel):
    user_question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("/query", status_code=501)
def answer_query(request: QueryRequest) -> dict:
    """Online pipeline: retrieve -> augment -> generate. Not yet built.

    Returns 501 rather than 404 deliberately. 404 would say "no such
    endpoint", which is false and would let a client conclude the feature
    was never planned; 501 says the route exists, the contract is this
    one, and the implementation is pending. The request body is still
    validated, so a caller integrating against it finds out today whether
    their payload is right.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "reason": "retrieval_not_implemented",
            "message": (
                "The online pipeline needs a vector index. The offline "
                "pipeline (POST /api/v1/index/run) produces anonymised, "
                "validated Documents; chunking and embedding are the next "
                "stage."
            ),
        },
    )
