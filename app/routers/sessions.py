"""Session-scoped, multi-turn estimation.

``POST /api/v1/sessions`` creates an empty conversational session.
``GET /api/v1/sessions/{session_id}`` returns a read-only snapshot of its
memory (summary, anchors, metadata, counters) for debugging and evals.
``POST /api/v1/sessions/{session_id}/estimate`` runs one turn of it: a
transcript plus optional attachments in, an updated ``EstimationResult`` and
``project_metadata`` out. Same error mapping as the single-shot endpoint
(``InputGuardrailViolation`` -> 400, anything else from the pipeline -> 502),
plus 404 when the session id doesn't exist (expired process restart, typo, or
a client that never called ``POST /sessions``).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_estimation_service, get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.services.estimation import EstimationService
from app.sessions import SessionEstimateResponse, SessionSnapshot, SessionStore

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["sessions"])


@router.post("/sessions")
def create_session(store: SessionStore = Depends(get_session_store)) -> dict:
    session = store.create()
    log.info("session_created", session_id=session.session_id)
    return {"session_id": session.session_id}


@router.get("/sessions/{session_id}", response_model=SessionSnapshot)
def get_session(
    session_id: str, store: SessionStore = Depends(get_session_store)
) -> SessionSnapshot:
    """Read-only X-ray of a session's memory.

    Exists so the CAG's three memory slots can be inspected from *outside*
    the process. The alternative — reaching into ``SessionStore`` from a
    test — only works in-process, and the evals that need this run over
    HTTP against the containerised service.

    No side effects: it must never run compression, refresh metadata or
    touch the LLM, or observing the system would change it.
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Create one with POST /api/v1/sessions.",
        )
    return SessionSnapshot.from_session(session)


@router.post("/sessions/{session_id}/estimate", response_model=SessionEstimateResponse)
async def estimate_turn(
    session_id: str,
    transcript: str = Form(..., min_length=10, max_length=80000),
    project_type: ProjectType | None = Form(None),
    detail_level: DetailLevel | None = Form(None),
    output_format: OutputFormat | None = Form(None),
    attachments: list[UploadFile] | None = File(default=None),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
) -> SessionEstimateResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, 
            detail="Session not found. Create one with POST /api/v1/sessions."
        )

    attachment_files = [(f.filename or "attachment", f.content_type, await f.read()) for f in attachments or []]

    log.info(
        "session_estimation_request_received",
        session_id=session_id,
        transcript_chars=len(transcript),
        attachments=len(attachment_files),
    )

    try:
        return service.estimate_in_session(
            session,
            transcript=transcript,
            attachment_files=attachment_files,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except InputGuardrailViolation as exc:
        log.info("session_estimation_blocked_by_input_guardrail", reason=exc.reason, message=exc.message)
        raise HTTPException(
            status_code=400, 
            detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except Exception as exc:
        log.error(
            "session_estimation_endpoint_error", error=str(exc)[:400], error_type=type(exc).__name__
        )
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc
