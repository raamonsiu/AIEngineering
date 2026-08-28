"""Cumulative summarizer for evicted conversation turns.

Invoked by ``CompressionPolicy`` whenever non-anchor turns fall off the
sliding window. Takes the previous running summary (may be empty) plus the
freshly-evicted messages and folds them into a new summary. The output
replaces the previous one, there is a single rolling summary per session,
not a chain of them.

Runs through the same ``LLMWrapper.complete_structured_with_messages``
primitive every other structured call in this service uses, routed to the
"compression" Router group (cheap model, own primary/fallback pair, see
``app/services/llm_wrapper.py``). On failure we keep the previous summary
intact, better to lose one compaction pass than to wipe state the
conversation has already committed to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from app.prompts.loader import render_conversation_summary_prompt
from app.sessions.models import Message

if TYPE_CHECKING:
    from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()


class _SummaryEnvelope(BaseModel):
    """Tiny structured wrapper so the LLM commits to a single string field,
    via Instructor for consistency with the rest of the stack rather than a
    plain free-text completion."""

    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    def __init__(self, *, llm_wrapper: "LLMWrapper") -> None:
        self.llm_wrapper = llm_wrapper

    def summarize(self, *, previous_summary: str | None, evicted: list[Message]) -> str:
        """Return the updated cumulative summary.

        On any LLM error we log and return ``previous_summary or ""`` so the
        session keeps running. Compression is best-effort; the sliding
        window guarantees the recent turns are always intact regardless.
        """
        if not evicted:
            return previous_summary or ""

        system_prompt, user_message = render_conversation_summary_prompt(
            previous_summary=previous_summary,
            evicted=[{"role": m.role, "content": m.content} for m in evicted],
        )

        try:
            envelope, meta = self.llm_wrapper.complete_structured_with_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=_SummaryEnvelope,
                model_name="compression",
                max_tokens=1000,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 — fail open, see module docstring
            log.warning(
                "summarizer_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )
            return previous_summary or ""

        log.info(
            "summarizer_completed",
            evicted_count=len(evicted),
            previous_chars=len(previous_summary or ""),
            new_chars=len(envelope.summary),
            model=meta.get("model"),
            latency_ms=meta.get("latency_ms"),
        )
        return envelope.summary
