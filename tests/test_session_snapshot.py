"""Tests for ``GET /api/v1/sessions/{id}`` — the read-only memory X-ray.

The interesting property is not that it returns 200: it's that it returns
the *contents* of the three memory slots (summary, anchors, metadata), and
that reading them does not mutate the session.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_estimation_service
from app.main import app
from app.services.estimation import EstimationService
from tests._session_test_helpers import FakeLLMWrapper


@pytest.fixture
def fake_wrapper():
    wrapper = FakeLLMWrapper()
    service = EstimationService(
        llm_wrapper=wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        prompt_version="v1",
        conversational_prompt_version="v3",
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    yield wrapper
    app.dependency_overrides.pop(get_estimation_service, None)


def test_unknown_session_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/sessions/does-not-exist").status_code == 404


def test_fresh_session_reports_empty_memory(client: TestClient) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()

    assert snapshot["session_id"] == session_id
    assert snapshot["message_count"] == 0
    assert snapshot["turn_count"] == 0
    assert snapshot["anchors_count"] == 0
    assert snapshot["summary_chars"] == 0
    assert snapshot["summary"] is None
    assert snapshot["anchors"] == []
    assert snapshot["last_turn"] is None
    assert snapshot["max_turns"] == get_settings().MAX_TURNS


def test_snapshot_exposes_slot_contents_not_just_counters(
    client: TestClient, fake_wrapper
) -> None:
    """The whole reason the endpoint exists: an eval must be able to ask
    "is the project name still in memory?", which a counter cannot answer."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "Project Aurora: a React dashboard over PostgreSQL."},
    )

    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()

    assert snapshot["turn_count"] == 1
    assert snapshot["message_count"] == 2
    assert snapshot["project_metadata"]["project_name"] == "Aurora"
    assert "React" in snapshot["project_metadata"]["mentioned_technologies"]
    assert snapshot["last_resolved_tier"] is not None


def test_snapshot_is_side_effect_free(client: TestClient, fake_wrapper) -> None:
    """Observing the system must not change it: no LLM call, no compression."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "A small internal tool to track equipment loans."},
    )
    calls_after_turn = len(fake_wrapper.calls)

    first = client.get(f"/api/v1/sessions/{session_id}").json()
    second = client.get(f"/api/v1/sessions/{session_id}").json()

    assert len(fake_wrapper.calls) == calls_after_turn
    assert first == second


def test_snapshot_reflects_compression_after_the_window_overflows(
    client: TestClient, fake_wrapper
) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    max_turns = get_settings().MAX_TURNS

    for i in range(max_turns + 2):
        client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": f"Turn {i}: please refine the estimate a little further."},
        )

    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()

    # The window is capped, and what fell out of it became summary text.
    assert snapshot["turn_count"] == max_turns
    assert snapshot["message_count"] == max_turns * 2
    assert snapshot["summary"] is not None
    assert snapshot["summary_chars"] == len(snapshot["summary"])
