"""Jinja2 loader for versioned prompt templates.

The on-disk layout is ``app/prompts/<use_case>/<version>/<role>.j2``. Versioning
is required from day one: switching prompts becomes a config change
(``PROMPT_VERSION`` in settings), not a code refactor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectMetadata, ProjectType

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
    """Render the system and user prompts for the estimation use case.

    Returns:
        A tuple ``(system_prompt, user_prompt)`` ready to be sent to the LLM as
        separate ``role: "system"`` and ``role: "user"`` messages.
    """
    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "project_metadata": None,
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
    project_metadata: ProjectMetadata,
    attachments_block: str = "",
    version: str = "v1",
) -> tuple[str, str]:
    """Render the system and user prompts for one turn of a session.

    Same system template as the single-shot flow, the ``<project_metadata>``
    block is what differs turn to turn, but a dedicated user template, since
    the session turn carries a transcript plus an optional attachments block
    instead of a single free-text ``description``.
    """
    system_context = {
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
        # Every field always present (no exclude_none): StrictUndefined would
        # raise on a template access to a key that got dropped for being None.
        "project_metadata": project_metadata.model_dump() if project_metadata else None,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**system_context)
    user = _env.get_template(f"estimation/{version}/session_user.j2").render(
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
    current_metadata: ProjectMetadata,
    user_turn: str,
    assistant_summary: str,
    version: str = "v1",
) -> str:
    """Render the extraction prompt for updating ``ProjectMetadata`` after a
    turn. A single free-standing prompt (no separate system/user split) since
    it drives one small, self-contained structured-output call."""
    return _env.get_template(f"estimation/{version}/metadata_extraction.j2").render(
        current_metadata_json=current_metadata.model_dump_json(),
        user_turn=user_turn,
        assistant_summary=assistant_summary,
    )
