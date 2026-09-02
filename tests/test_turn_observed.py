"""Tests for the ``turn_observed`` aggregated event (Session 6, Block 1).

The contract under test is the event's *shape* and the two properties that
make it usable as a dataset: a turn index that keeps counting after the
sliding window stops growing, and totals that include the turn's side LLM
calls, not just the estimator's.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_estimation_service
from app.main import app
from app.services.estimation import EstimationService
from tests._session_test_helpers import FakeLLMWrapper, make_test_pdf

REQUIRED_FIELDS = {
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
}


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


def _run_turn(client: TestClient, session_id: str, transcript: str, **kwargs) -> dict:
    response = client.post(
        f"/api/v1/sessions/{session_id}/estimate", data={"transcript": transcript}, **kwargs
    )
    assert response.status_code == 200, response.text
    return client.get(f"/api/v1/sessions/{session_id}").json()["last_turn"]


def test_event_carries_all_thirteen_required_fields(client: TestClient, fake_wrapper) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    observed = _run_turn(client, session_id, "A React dashboard for tracking equipment loans.")

    assert REQUIRED_FIELDS <= set(observed)
    assert observed["turn_index"] == 1
    assert observed["session_id"] == session_id
    assert observed["cache_hit_kind"] == "none"
    assert observed["last_resolved_tier"] is not None


def test_turn_index_keeps_counting_after_the_window_caps(client: TestClient, fake_wrapper) -> None:
    """The x-axis of every degradation curve. ``len(history)`` saturates at
    ``max_turns``; ``turn_index`` must not."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    max_turns = get_settings().MAX_TURNS
    total_turns = max_turns + 3

    indices, windows = [], []
    for i in range(total_turns):
        observed = _run_turn(client, session_id, f"Turn {i}: refine the estimate a little further.")
        indices.append(observed["turn_index"])
        windows.append(observed["messages_in_window"])

    assert indices == list(range(1, total_turns + 1))
    assert windows[-1] == max_turns * 2  # the window saturated
    assert max(windows) == max_turns * 2


def test_totals_include_the_turns_side_calls_not_just_the_estimator(
    client: TestClient, fake_wrapper
) -> None:
    """The estimator call alone reports 500 in / 120 out. The metadata
    extractor adds 80 / 20 on every turn, and the whole point of the
    per-turn collector is that those land in the same row."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    observed = _run_turn(client, session_id, "A React dashboard over PostgreSQL.")

    assert observed["llm_calls"] == 2  # estimator + metadata extractor
    assert observed["tokens_in"] == 500 + 80
    assert observed["tokens_out"] == 120 + 20
    assert observed["cost_usd"] == pytest.approx(0.0001 + 0.00001)


def test_a_compressing_turn_shows_the_extra_summarizer_call(
    client: TestClient, fake_wrapper
) -> None:
    """Cost per turn is not flat, and the collector must show why: once the
    window overflows, a third call joins every turn."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    max_turns = get_settings().MAX_TURNS

    first = _run_turn(client, session_id, "Turn 0: an internal tool to track equipment loans.")
    for i in range(1, max_turns):
        _run_turn(client, session_id, f"Turn {i}: refine the estimate a little further.")
    compressing = _run_turn(
        client, session_id, f"Turn {max_turns}: this one pushes the window over the cap."
    )

    assert first["llm_calls"] == 2
    assert compressing["llm_calls"] == 3  # + summarizer
    assert compressing["cost_usd"] > first["cost_usd"]
    assert compressing["summary_chars"] > 0


def test_attachment_chars_are_counted_into_the_enriched_transcript(
    client: TestClient, fake_wrapper
) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    transcript = "Estimate this project using the attached brief."
    pdf = make_test_pdf("Requirements brief for the Aurora platform rollout.")

    observed = _run_turn(
        client,
        session_id,
        transcript,
        files={"attachments": ("brief.pdf", pdf, "application/pdf")},
    )

    assert observed["attachments_count"] == 1
    assert observed["attachments_total_chars"] > 0
    assert observed["enriched_transcript_chars"] == len(transcript) + observed["attachments_total_chars"]


def test_no_attachment_reports_zero_attachment_chars(client: TestClient, fake_wrapper) -> None:
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    transcript = "A small internal tool to track equipment loans."

    observed = _run_turn(client, session_id, transcript)

    assert observed["attachments_count"] == 0
    assert observed["attachments_total_chars"] == 0
    assert observed["enriched_transcript_chars"] == len(transcript)
