"""In-process data structures for the conversational estimator.

Design notes
------------
- ``ConversationHistory.to_messages()`` returns the LLM-ready ``messages``
  array. The system prompt is NOT stored here: it is re-rendered fresh from
  the session's current ``ProjectMetadata`` on every turn and prepended by
  the caller, so "always preserve the system prompt" is automatic rather
  than a truncation special case.
- ``ProjectMetadata`` fields are all optional: an empty metadata block is a
  legitimate first-turn state. ``merge_with`` is the deterministic
  Python-side merge that survives a turn where the LLM extractor forgets to
  restate a fact, scalar overwrite (only when the update provides a
  non-null value) and list-union for the list fields. This is a Python
  invariant, not a prompt instruction: the extractor is asked to return only
  what's new/changed, and this method is what actually preserves the rest.
- ``ConversationHistory`` no longer trims itself on ``append`` (Session 5):
  once anchors exist, deciding whether an evicted turn is disposable or
  must be rescued verbatim requires inspecting it first, which ``append``
  has no business doing. That decision, plus folding disposable turns into
  ``summary``, is ``app.sessions.compression.CompressionPolicy``'s job,
  called explicitly by the service after every ``append``. Keeping the data
  structure dumb means there is exactly one place where the LLM is asked to
  decide what to forget.
- Storage is a plain process-local dict (see ``SessionStore``), no
  database, no Redis. A session is short-lived working memory for one active
  conversation, not a system of record, and this phase of the project is
  about the CAG architecture (separating history from memory), not about
  surviving a restart.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.estimation import (
    CallMeta,
    DetailLevel,
    EstimationResult,
    OutputFormat,
    ProjectType,
)

DEFAULT_MAX_TURNS = 6

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One message in the conversation history. The system prompt is NOT a
    ``Message``, it lives outside the history because it is re-rendered
    each turn from the current ``ProjectMetadata``."""

    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationHistory(BaseModel):
    """A sliding window of user/assistant pairs, augmented with a cumulative
    summary and anchored turns (Session 5's hybrid compression strategy).

    Three storage slots:

    - ``messages``: the recent sliding window (last ``max_turns`` pairs).
      ``max_turns`` counts pairs; the cap is enforced by
      ``app.sessions.compression.CompressionPolicy``, not by this class (see
      the module docstring).
    - ``anchors``: turns the ``AnchorDetector`` flagged as durable
      commitments (signed contract, frozen scope, locked budget, legal/
      compliance mention). They live outside the sliding window and are
      never evicted by it.
    - ``summary``: a free-text cumulative summary of older non-anchor turns
      the ``CompressionPolicy`` has already folded away.
    """

    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1)
    messages: list[Message] = Field(default_factory=list)
    anchors: list[Message] = Field(default_factory=list)
    summary: str | None = Field(default=None)

    def append(self, *, user: str, assistant: str) -> None:
        """Add one turn (user message + assistant message).

        Does NOT trim: whether an overflowing turn is disposable or must be
        rescued as an anchor is ``CompressionPolicy.apply``'s call, and it
        needs to see the turn before it decides. The caller runs that policy
        right after this append (see ``EstimationService.estimate_in_session``).
        """
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))

    def to_messages(self) -> list[dict[str, str]]:
        """Return the ``messages`` array ready to splice into the LLM call,
        excluding the system prompt (the caller prepends that fresh each
        turn).

        Order is intentional: the summary anchors the model in the early
        conversation, the anchors carry irreducible commitments verbatim,
        and the recent window provides turn-by-turn context.

            [summary_envelope?] + anchors_in_order + recent_sliding_window
        """
        out: list[dict[str, str]] = []
        if self.summary:
            # Wrapped as a synthetic user message, and visually marked, so
            # the model treats it as prior context rather than an
            # instruction from the current turn.
            out.append(
                {
                    "role": "user",
                    "content": (
                        "[Earlier conversation summary — the recent turns "
                        "below are the live thread]\n" + self.summary
                    ),
                }
            )
        for anchor in self.anchors:
            out.append({"role": anchor.role, "content": anchor.content})
        out.extend({"role": m.role, "content": m.content} for m in self.messages)
        return out

    def __len__(self) -> int:
        return len(self.messages) // 2


class ProjectMetadata(BaseModel):
    """Facts about the project under discussion, kept across turns.

    All fields are optional: on the first turn nothing has been established
    yet. The metadata extractor (``app/sessions/metadata_extractor.py``)
    populates them turn by turn, returning only what's new so ``merge_with``
    can combine it deterministically with what's already known.
    """

    project_name: str | None = Field(default=None, max_length=120)
    assumed_team_size: int | None = Field(default=None, ge=1, le=500)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=2000)
    explicit_constraints: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
            and not self.explicit_constraints
            and not self.rejected_options
        )

    def merge_with(self, update: "ProjectMetadata") -> "ProjectMetadata":
        """Return a new metadata where non-null fields from ``update`` win,
        and the list fields are unioned rather than replaced.

        This is the safety net for the fact the extractor prompt cannot
        guarantee on its own: an LLM is non-deterministic, and a turn where
        it forgets to restate ``project_name`` must NOT erase it. Scalar
        overwrite-if-present + list-union means a forgotten field simply
        falls back to what was already known, instead of regressing.
        """
        merged_tech = list(self.mentioned_technologies)
        seen_tech = {t.lower() for t in merged_tech}
        for tech in update.mentioned_technologies:
            if tech.lower() not in seen_tech:
                merged_tech.append(tech)
                seen_tech.add(tech.lower())

        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=merged_tech,
            agreed_scope=update.agreed_scope or self.agreed_scope,
            explicit_constraints=_union(self.explicit_constraints, update.explicit_constraints),
            rejected_options=_union(self.rejected_options, update.rejected_options),
        )


def _union(existing: list[str], new: list[str]) -> list[str]:
    """Append items from ``new`` that aren't already in ``existing`` (exact
    match, these are free-text sentences, not short tokens, so unlike
    technologies a case-insensitive fold would be more likely to merge two
    genuinely different constraints than to catch a real duplicate)."""
    merged = list(existing)
    seen = set(merged)
    for item in new:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


class Session(BaseModel):
    """A conversational estimation session, held in the process's memory.

    The typed selectors (``project_type``, ``detail_level``, ``output_format``)
    are optional on each turn of the multi-turn endpoint but remembered here:
    set once (or defaulted on creation) and only overwritten when a turn
    explicitly supplies a new value, this keeps the "typed request, not
    free chat" contract from the single-shot endpoint.

    ``last_resolved_tier``/``last_tier_rule`` cache the audience tier
    (see ``app.sessions.tier_resolver``) resolved for the most recent turn,
    purely so it can be echoed back on ``SessionEstimateResponse`` without
    re-running the resolver.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    project_metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    project_type: ProjectType = ProjectType.WEB_SAAS
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


class AttachmentReport(BaseModel):
    """What happened when processing one uploaded attachment."""

    filename: str
    method: str  # "pypdf" | "pymupdf" | "docx" | "llm_fallback" | "too_long" | "failed"
    ok: bool
    note: str | None = None


class SessionEstimateResponse(BaseModel):
    """Response for a multi-turn, session-scoped estimation. Same shape as
    ``EstimationResponse`` plus the session id, the ``project_metadata`` as
    it stands after this turn, a per-attachment processing report, and the
    audience tier resolved for this turn, so the client can render memory,
    attachment handling and tier framing without a second call."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    meta: CallMeta = Field(default_factory=CallMeta)
    session_id: str
    project_metadata: ProjectMetadata
    attachments: list[AttachmentReport] = Field(default_factory=list)
    resolved_tier: str
    tier_rule: str
