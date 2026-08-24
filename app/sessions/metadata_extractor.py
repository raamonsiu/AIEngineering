"""Second-pass LLM call that extracts ``ProjectMetadata`` from each turn.

Runs AFTER the main estimation succeeds. The prompt asks the model for only
the facts that are new or changed in THIS turn (not the full metadata object), ``ProjectMetadata.merge_with`` is what actually combines that with what was
already known. This split matters: an LLM is non-deterministic, so a call
that asked for "the full updated object" could silently drop a fact it forgot
to restate. Asking for a delta and merging deterministically in Python means
a forgotten field just falls back to the previous value instead of
regressing.

The call runs against a dedicated, usually-cheaper model/fallback pair
(``METADATA_EXTRACTOR_MODEL`` / ``METADATA_EXTRACTOR_FALLBACK_MODEL``) via the
same ``LLMWrapper.complete_structured_with_messages`` primitive used for the
main estimation, it still goes through the Router, so this side call keeps
the provider-fallback guarantee too, not just the main call.

If extraction fails for any reason (LLM error, validator exhausted retries),
we log the failure and return the previous metadata unchanged. The
conversation keeps working, losing one turn's metadata refresh is
acceptable; losing the conversation is not.
"""

from __future__ import annotations

import structlog

from app.prompts.loader import render_metadata_extraction_prompt
from app.schemas.estimation import EstimationResult
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import ProjectMetadata

log = structlog.get_logger()


def update_metadata(
    *,
    previous: ProjectMetadata,
    transcript: str,
    result: EstimationResult,
    llm_wrapper: LLMWrapper,
) -> ProjectMetadata:
    """Run the extractor and return ``previous.merge_with(extracted)``.

    On failure: log and return ``previous`` unchanged.
    """
    prompt = render_metadata_extraction_prompt(
        current_metadata_json=previous.model_dump_json(),
        previous_is_empty=previous.is_empty(),
        user_turn=transcript,
        assistant_content=result.model_dump_json(),
    )

    try:
        extracted, meta = llm_wrapper.complete_structured_with_messages(
            messages=[{"role": "user", "content": prompt}],
            response_model=ProjectMetadata,
            model_name="metadata_extractor",
            max_tokens=500,
            max_retries=2,
        )
    except Exception as exc:  # noqa: BLE001 - fail open, see module docstring
        log.warning(
            "metadata_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return previous

    merged = previous.merge_with(extracted)
    log.info(
        "metadata_extraction_completed",
        model=meta.get("model"),
        latency_ms=meta.get("latency_ms"),
        project_name=merged.project_name,
        team_size=merged.assumed_team_size,
        tech_count=len(merged.mentioned_technologies),
    )
    return merged
