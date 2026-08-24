"""Shared test doubles for the session test files. Not a test module itself
(leading underscore so pytest doesn't try to collect it) - just the fake LLM
wrapper and a tiny real-PDF builder reused across
``test_sessions_metadata.py``, ``test_sessions_attachments.py`` and
``test_sessions_window.py``.
"""

from __future__ import annotations

import pymupdf

from app.schemas.estimation import EstimationDraft
from app.sessions.models import ProjectMetadata


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


class FakeLLMWrapper:
    """Records every call so tests can assert on what the service built,
    without making a real network call. One method covers BOTH the main
    estimation call and the metadata-extraction call (dispatched by
    ``response_model``), the same split ``EstimationService`` actually uses
    via ``model_name="metadata_extractor"``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete_structured_with_messages(
        self, *, messages, response_model, model_name="estimator", max_tokens=4000, max_retries=6
    ):
        self.calls.append(
            {"messages": messages, "response_model": response_model, "model_name": model_name}
        )
        text_blob = " ".join(_flatten_content(m["content"]) for m in messages).lower()

        if response_model is EstimationDraft:
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

        if response_model is ProjectMetadata:
            # A stand-in extractor: scans the whole prompt (which embeds the
            # latest turn) for known keywords. It doesn't need to be
            # delta-aware to test accumulation correctly, ProjectMetadata's
            # real merge_with() is what's actually under test, and merging
            # is idempotent for repeated facts.
            delta = ProjectMetadata()
            if "aurora" in text_blob:
                delta.project_name = "Aurora"
            for keyword, label in [("react", "React"), ("postgresql", "PostgreSQL"), ("node", "Node")]:
                if keyword in text_blob:
                    delta.mentioned_technologies = [*delta.mentioned_technologies, label]
            meta = {"model": "gpt-4o-mini", "provider": "openai", "cost_usd": 0.00001, "latency_ms": 5}
            return delta, meta

        raise AssertionError(f"unexpected response_model: {response_model}")


def make_test_pdf(text: str) -> bytes:
    """A minimal, real PDF with an actual text layer, generated with
    PyMuPDF and round-tripped through pypdf in the extraction pipeline, so
    this exercises the real Camino B code path, not a mock of it."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data
