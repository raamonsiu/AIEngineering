"""Jinja2 loader for versioned prompt templates.

The on-disk layout is ``app/prompts/<use_case>/<version>/<role>.j2``. Versioning
is required from day one: switching prompts becomes a config change
(``PROMPT_VERSION`` / ``CONVERSATIONAL_PROMPT_VERSION`` in settings), not a
code refactor.

This module takes primitives (dicts, JSON strings, booleans, enum values) for
anything domain-shaped rather than importing the domain models themselves
(``ProjectMetadata`` and friends live in ``app.sessions``), a templating
utility has no reason to depend on the session/estimation domain, and callers
already have the model instance to hand when they call in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

log = structlog.get_logger()

_BASE_DIR = Path(__file__).resolve().parent

_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
    keep_trailing_newline=True,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Render the system and user prompts for the single-shot estimation use
    case.

    Returns:
        A tuple ``(system_prompt, user_prompt)`` ready to be sent to the LLM as
        separate ``role: "system"`` and ``role: "user"`` messages.
    """
    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)

    # Content hashes (not full text) so production logs can confirm which exact
    # rendered prompt a given call used, without dumping the user's description
    # into the log stream.
    log.info(
        "prompt_rendered",
        prompt_version=version,
        system_hash=hashlib.sha256(system.encode("utf-8")).hexdigest()[:12],
        user_hash=hashlib.sha256(user.encode("utf-8")).hexdigest()[:12],
    )
    return system, user


def render_session_prompt(
    *,
    transcript: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    project_metadata: dict,
    metadata_is_empty: bool,
    tier: str = "default",
    attachments_block: str = "",
    version: str = "v2",
) -> tuple[str, str]:
    """Render the system and user prompts for one turn of a conversational
    session. Its own dedicated version (``v2`` by default) so the
    conversational prompt can evolve independently of the single-shot one.

    ``tier`` is the audience tier resolved for this turn (see
    ``app.sessions.tier_resolver``), passed as its plain string value: only
    the ``v3`` template's ``<audience>`` block reads it, ``v2`` simply
    ignores the extra context key.
    """
    system_context = {
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
        "project_metadata": project_metadata,
        "metadata_is_empty": metadata_is_empty,
        "tier": tier,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**system_context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(
        transcript=transcript,
        attachments_block=attachments_block,
        project_type=project_type.value,
    )

    log.info(
        "prompt_rendered",
        prompt_version=version,
        kind="session",
        system_hash=hashlib.sha256(system.encode("utf-8")).hexdigest()[:12],
        user_hash=hashlib.sha256(user.encode("utf-8")).hexdigest()[:12],
    )
    return system, user


def render_metadata_extraction_prompt(
    *,
    current_metadata_json: str,
    previous_is_empty: bool,
    user_turn: str,
    assistant_content: str,
    version: str = "v1",
) -> str:
    """Render the extraction prompt for updating ``ProjectMetadata`` after a
    turn. A single free-standing prompt (no separate system/user split) since
    it drives one small, self-contained structured-output call."""
    return _env.get_template(f"estimation/{version}/metadata_extraction.j2").render(
        current_metadata_json=current_metadata_json,
        previous_is_empty=previous_is_empty,
        user_turn=user_turn,
        assistant_content=assistant_content,
    )


def render_conversation_summary_prompt(
    *,
    previous_summary: str | None,
    evicted: list[dict[str, str]],
    version: str = "v1",
) -> tuple[str, str]:
    """Render the system and user prompts for one cumulative-summary pass
    (Session 5 compression). ``evicted`` is the plain ``[{"role", "content"}]``
    list of the turns falling off the sliding window, same shape
    ``ConversationHistory.to_messages()`` already produces elsewhere, so the
    caller (``CumulativeSummarizer``) never has to hand this module a
    ``Message`` instance.
    """
    system = _env.get_template(f"conversation_summary/{version}/system.j2").render()
    user = _env.get_template(f"conversation_summary/{version}/user.j2").render(
        previous_summary=previous_summary,
        evicted=evicted,
    )

    log.info(
        "prompt_rendered",
        prompt_version=version,
        kind="conversation_summary",
        system_hash=hashlib.sha256(system.encode("utf-8")).hexdigest()[:12],
        user_hash=hashlib.sha256(user.encode("utf-8")).hexdigest()[:12],
    )
    return system, user
