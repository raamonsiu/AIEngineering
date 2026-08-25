"""Integration tests for attachment handling in the session flow: content
reaching the LLM, and the abstracted (word-count, not token) length limit.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_estimation_service
from app.main import app
from app.services.estimation import EstimationService
from tests._session_test_helpers import FakeLLMWrapper, make_test_pdf


@pytest.fixture
def fake_wrapper():
    wrapper = FakeLLMWrapper()
    service = EstimationService(
        llm_wrapper=wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        prompt_version="v1",
        conversational_prompt_version="v2",
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    yield wrapper
    app.dependency_overrides.pop(get_estimation_service, None)


def test_attachment_content_influences_the_estimate(client: TestClient, fake_wrapper) -> None:
    pdf_bytes = make_test_pdf(
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


def test_oversized_attachment_is_reported_without_blocking_the_estimate(
    client: TestClient,
) -> None:
    """A too-long attachment must not 502 the request — it's dropped and
    reported, with a human-readable (word count, not token count) reason."""
    wrapper = FakeLLMWrapper()
    service = EstimationService(
        llm_wrapper=wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        prompt_version="v1",
        conversational_prompt_version="v2",
        max_attachment_words=5,
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    try:
        session_id = client.post("/api/v1/sessions").json()["session_id"]
        pdf_bytes = make_test_pdf("one two three four five six seven eight nine ten")

        response = client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": "A small internal tool to track equipment loans across teams."},
            files={"attachments": ("long.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        report = response.json()["attachments"][0]
        assert report["ok"] is False
        assert report["method"] == "too_long"
        assert "words" in report["note"]
        assert "5" in report["note"]
    finally:
        app.dependency_overrides.pop(get_estimation_service, None)
