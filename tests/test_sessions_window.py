"""Integration tests for the session endpoints' basic mechanics: creating a
session, 404 on an unknown one, and the sliding window never growing past
``MAX_TURNS`` worth of history in the messages actually sent to the LLM.
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
        conversational_prompt_version="v2",
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


def test_history_window_never_exceeds_max_turns(client: TestClient, fake_wrapper) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    max_turns = get_settings().MAX_TURNS

    for i in range(max_turns + 2):
        response = client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": f"Turn number {i}: please refine the estimate a bit more."},
        )
        assert response.status_code == 200

    estimator_calls = [c for c in fake_wrapper.calls if c["model_name"] == "estimator"]
    for call in estimator_calls:
        # system prompt + at most max_turns*2 prior turns + this turn's message.
        assert len(call["messages"]) <= 1 + 2 * max_turns + 1

    # By the last call the window is actually full, not just under the cap.
    assert len(estimator_calls[-1]["messages"]) == 1 + 2 * max_turns + 1
