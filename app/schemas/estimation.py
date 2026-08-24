"""Request and response models for the estimation endpoint.

Contract: a typed, form-style request maps to a typed, validated
``EstimationResult`` (structured output via Instructor + Pydantic). Two model
validators enforce business rules the LLM cannot break:

1. The cost of all phases must sum to ``total_cost_eur``.
2. Low-confidence answers (< 30%) must declare it explicitly by starting the
   summary with ``"Out of scope:"``.

When the LLM violates a validator, Instructor re-prompts the model with the
``ValueError`` message until it agrees (up to ``max_retries`` attempts).
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    """Typed payload sent by the Streamlit form or any other client."""

    description: str = Field(
        min_length=20,
        max_length=80000,
        description="Free-text description or transcription of the project to estimate.",
    )
    project_type: ProjectType = Field(description="Coarse-grained project category.")
    detail_level: DetailLevel = Field(description="How deep the estimation should go.")
    output_format: OutputFormat = Field(description="Shape of the rendered estimation.")


# --- Structured response ----------------------------------------------------

OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """One phase in the breakdown of an estimation."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationDraft(BaseModel):
    """What the LLM is actually asked to fill in: everything except the totals.

    Why no total fields here, measured, not assumed. Asking ``gpt-4o-mini`` for
    a grand total alongside the phases fails reliably: in a live run against
    this exact prompt it burned all 7 Instructor attempts (~24k tokens) without
    ever making ``sum(phases) == total_cost_eur``, drifting further on each
    retry, and the request 502'd. Field ordering (phases before totals) and an
    explicit "compute the sum" instruction in the prompt were not enough.

    So the arithmetic is done in Python instead (see
    ``services/estimation.py``), which removes that entire class of failure at
    zero cost. ``EstimationResult`` keeps its ``phases_sum_matches_total``
    validator as a genuine invariant assertion over the computed value.

    The low-confidence rule DOES live here, because this is the model Instructor
    validates and retries against : putting it only on ``EstimationResult``
    would mean the retry loop never sees a violation.
    """

    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationDraft":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self


class EstimationResult(BaseModel):
    """Structured estimation returned to the client. The two validators below
    are the business rules that must hold whatever produced the object.

    Field order is deliberate: ``phases`` comes BEFORE the totals, so if this
    model is ever handed to an LLM directly it commits to the per-phase numbers
    first (autoregressive generation) rather than picking a round total and
    back-fitting phases to it.
    """

    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @classmethod
    def from_draft(cls, draft: EstimationDraft) -> "EstimationResult":
        """Build the result by summing the draft's phases in Python."""
        return cls(
            summary=draft.summary,
            confidence_pct=draft.confidence_pct,
            phases=draft.phases,
            total_duration_weeks=sum(p.duration_weeks for p in draft.phases),
            total_cost_eur=sum(p.cost_eur for p in draft.phases),
        )

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> "EstimationResult":
        phase_sum = sum(p.cost_eur for p in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(
                f"phases sum ({phase_sum} EUR) does not match total_cost_eur "
                f"({self.total_cost_eur} EUR); adjust either the phases or the total"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self


class CallMeta(BaseModel):
    """What this particular call cost. On a cache hit nothing was spent, so
    ``cost_usd`` is 0.0 and ``model``/``provider`` describe whichever
    deployment originally produced the cached result."""

    model: str | None = None
    provider: str | None = None
    cost_usd: float = 0.0
    latency_ms: int = 0


class EstimationResponse(BaseModel):
    """Wraps the validated result, the prompt version that produced it, and
    whether it came from a cache (exact or semantic)."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    meta: CallMeta = Field(default_factory=CallMeta)


class ProjectMetadata(BaseModel):
    """Distilled, durable facts about the project under discussion in a
    session, separate from the raw turn history, which is subject to the
    sliding window and can be truncated.

    Populated by a dedicated LLM extraction call after each turn
    (``LLMWrapper.extract_metadata``) rather than a regex heuristic: the
    conversation here is free-form and bilingual (ES/EN), the case the course
    material itself flags as tipping the balance away from heuristics.
    """

    project_name: str | None = None
    assumed_team_size: int | None = Field(default=None, ge=1, le=500)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None
    explicit_constraints: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)


class AttachmentReport(BaseModel):
    """What happened when processing one uploaded attachment."""

    filename: str
    method: str  # "pypdf" | "pymupdf" | "docx" | "llm_fallback" | "failed"
    ok: bool
    note: str | None = None


class SessionEstimateResponse(BaseModel):
    """Response for a multi-turn, session-scoped estimation. Same shape as
    ``EstimationResponse`` plus the session id, the ``project_metadata`` as it
    stands after this turn, and a per-attachment processing report, so the
    client can render memory and attachment handling without a second call."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    meta: CallMeta = Field(default_factory=CallMeta)
    session_id: str
    project_metadata: ProjectMetadata
    attachments: list[AttachmentReport] = Field(default_factory=list)
