"""Integration test: project_metadata accumulates correctly across turns of
the same session. Only the LLM call is faked (``FakeLLMWrapper``), session
creation/lookup, prompt rendering and the real ``ProjectMetadata.merge_with``
all run for real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

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
    # is exactly what project_metadata (and its deterministic merge) is for.
    assert metadata_after_second_turn["project_name"] == "Aurora"
    assert set(metadata_after_second_turn["mentioned_technologies"]) == {
        "React",
        "PostgreSQL",
        "Node",
    }


def test_prompt_version_reported_is_the_conversational_one(
    client: TestClient, fake_wrapper
) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "A small internal tool to track equipment loans."},
    )

    assert response.json()["prompt_version"] == "v2"
