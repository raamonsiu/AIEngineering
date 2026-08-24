"""Integration tests for the multi-turn, session-scoped estimation flow.

Only the LLM call itself is faked (``FakeLLMWrapper``), everything else runs
for real: session creation/lookup (``SessionStore``), the sliding-window
history, the Jinja2 prompt rendering, and the attachment extraction pipeline
(a real PDF is generated with PyMuPDF and parsed back with pypdf). This is
deliberately a lower-level fake than ``test_estimate_endpoint.py``'s
``FakeEstimationService``, faking the whole service would hide exactly the
logic these tests exist to check (window truncation, metadata persistence,
attachment handling).
"""

from __future__ import annotations

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_estimation_service
from app.main import app
from app.schemas.estimation import EstimationDraft
from app.services.estimation import EstimationService


class FakeLLMWrapper:
    """Records every call so tests can assert on what the service built,
    without making a real network call."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.metadata_calls: list[str] = []

    def complete_structured_with_messages(
        self, *, messages, response_model, max_tokens=4000, max_retries=6
    ):
        self.calls.append(messages)
        text_blob = " ".join(_flatten_content(m["content"]) for m in messages).lower()
        confidence = 90 if "biometric" in text_blob else 65
        draft = EstimationDraft(
            summary="Estimated project based on the conversation so far.",
            confidence_pct=confidence,
            phases=[
                {
                    "name": "Implementation",
                    "duration_weeks": 4,
                    "cost_eur": 8_000,
                    "summary": "Core build based on the transcript.",
                }
            ],
        )
        meta = {"model": "gpt-4o-mini", "provider": "openai", "cost_usd": 0.0001, "latency_ms": 10}
        return draft, meta

    def extract_metadata(self, *, current, user_turn, assistant_summary, max_retries=2):
        self.metadata_calls.append(user_turn)
        updated = current.model_copy(deep=True)
        lowered = user_turn.lower()
        if "aurora" in lowered and updated.project_name is None:
            updated.project_name = "Aurora"
        for keyword, label in [("react", "React"), ("postgresql", "PostgreSQL"), ("node", "Node")]:
            if keyword in lowered and label not in updated.mentioned_technologies:
                updated.mentioned_technologies = [*updated.mentioned_technologies, label]
        return updated


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


def _make_test_pdf(text: str) -> bytes:
    """A minimal, real PDF with an actual text layer, generated with
    PyMuPDF and round-tripped through pypdf in the extraction pipeline, so
    this exercises the real Camino B code path, not a mock of it."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def fake_wrapper():
    wrapper = FakeLLMWrapper()
    service = EstimationService(
        llm_wrapper=wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        prompt_version="v1",
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    yield wrapper
    app.dependency_overrides.pop(get_estimation_service, None)


def test_post_sessions_returns_a_session_id(client: TestClient) -> None:
    response = client.post("/api/v1/sessions")
    assert response.status_code == 200
    assert "session_id" in response.json()


def test_estimate_with_unknown_session_returns_404(client: TestClient, fake_wrapper) -> None:
    response = client.post(
        "/api/v1/sessions/does-not-exist/estimate",
        data={"transcript": "A small internal tool to track equipment loans."},
    )
    assert response.status_code == 404


def test_session_metadata_persists_and_accumulates_across_turns(
    client: TestClient, fake_wrapper
) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    first = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "The project is called Aurora, built with React and PostgreSQL."},
    )
    assert first.status_code == 200
    metadata = first.json()["project_metadata"]
    assert metadata["project_name"] == "Aurora"
    assert set(metadata["mentioned_technologies"]) == {"React", "PostgreSQL"}

    second = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "Let's also add a Node-based background worker for exports."},
    )
    assert second.status_code == 200
    metadata_after_second_turn = second.json()["project_metadata"]

    # Facts from turn 1 survive even though turn 2 never repeats them, this
    # is exactly what project_metadata is for, as opposed to raw history.
    assert metadata_after_second_turn["project_name"] == "Aurora"
    assert set(metadata_after_second_turn["mentioned_technologies"]) == {
        "React",
        "PostgreSQL",
        "Node",
    }


def test_attachment_content_influences_the_estimate(client: TestClient, fake_wrapper) -> None:
    pdf_bytes = _make_test_pdf(
        "Additional requirement: the app must support biometric login and "
        "offline-first sync, as detailed in this specification."
    )
    transcript = "A small internal tool to track equipment loans across teams."

    session_without = client.post("/api/v1/sessions").json()["session_id"]
    without_attachment = client.post(
        f"/api/v1/sessions/{session_without}/estimate", data={"transcript": transcript}
    )

    session_with = client.post("/api/v1/sessions").json()["session_id"]
    with_attachment = client.post(
        f"/api/v1/sessions/{session_with}/estimate",
        data={"transcript": transcript},
        files={"attachments": ("spec.pdf", pdf_bytes, "application/pdf")},
    )

    assert without_attachment.status_code == 200
    assert with_attachment.status_code == 200

    # The fake LLM's confidence depends on whether "biometric" reached it,
    # a stand-in for "the document's content changed what got estimated".
    assert without_attachment.json()["result"]["confidence_pct"] == 65
    assert with_attachment.json()["result"]["confidence_pct"] == 90

    attachments = with_attachment.json()["attachments"]
    assert len(attachments) == 1
    assert attachments[0]["ok"] is True
    assert attachments[0]["method"] == "pypdf"


def test_history_window_never_exceeds_max_turns(client: TestClient, fake_wrapper) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    max_turns = get_settings().MAX_TURNS

    for i in range(max_turns + 2):
        response = client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": f"Turn number {i}: please refine the estimate a bit more."},
        )
        assert response.status_code == 200

    for messages in fake_wrapper.calls:
        # system prompt + at most max_turns*2 prior turns + this turn's message.
        assert len(messages) <= 1 + 2 * max_turns + 1

    # By the last call the window is actually full, not just under the cap.
    assert len(fake_wrapper.calls[-1]) == 1 + 2 * max_turns + 1
