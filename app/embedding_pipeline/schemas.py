"""The contracts of the embedding pipeline: budget in, embedded chunk out.

Three groups of models live here, and the boundaries between them are the
pipeline's seams:

- ``Budget`` / ``BudgetComponent`` / ``ClientMetadata`` describe the input
  as it arrives: a historical budget, already normalised upstream.
- ``Chunk`` is what the chunker produces. It is the unit that gets
  embedded, so everything that must influence the vector has to be inside
  ``text``, and everything that must only *filter* results has to be
  outside it, in ``metadata``.
- ``EmbeddedChunk`` is ``Chunk`` plus the vector. Keeping them as two
  types rather than one with an optional field means the chunker cannot
  accidentally return something half-embedded, and the embedder's
  signature says exactly what it adds.

Nothing here persists: Session 07 returns the vectors over HTTP and
forgets them. Persistence is Session 08's problem (PostgreSQL + pgvector),
which is why there is no id, no timestamp and no revision field.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# The separator between a budget id and a component id inside a chunk id.
# Declared here rather than inlined in the chunker because the validators
# below have to reject ids that would make the composite ambiguous.
CHUNK_ID_SEPARATOR = "::"

# Closed because the four sectors are the universe the corpus covers, and
# an unexpected value is far more likely to be a typo than a new market.
# A 422 naming the accepted values is a better outcome than a chunk whose
# ``client_sector`` filter silently never matches anything.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]

# Normalisations applied before the Literal check. These are spellings of
# the *same* sector, not aliases for new ones: "fintech" and "finance" are
# the same filter, and rejecting one of them would be pedantry.
_SECTOR_ALIASES = {
    "fintech": "finance",
    "financial": "finance",
    "banking": "finance",
    "e-commerce": "ecommerce",
    "e_commerce": "ecommerce",
    "retail": "ecommerce",
    "health": "healthcare",
    "medical": "healthcare",
    "manufacturing": "industrial",
}

Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    """Who the budget was written for. Carries no personal data: the client
    is a company, and the people involved were already pseudonymised by the
    Session 06 ingest pipeline before anything reached this module."""

    name: str = Field(min_length=1, max_length=120)
    sector: Sector
    country: str = Field(
        pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 country code, e.g. ES."
    )

    @field_validator("sector", mode="before")
    @classmethod
    def normalise_sector(cls, value: Any) -> Any:
        """Fold known spellings onto the canonical sector before the Literal
        is checked. Runs ``mode="before"`` because by the time Pydantic
        validates the Literal the value is already either right or an error.
        """
        if isinstance(value, str):
            folded = value.strip().lower()
            return _SECTOR_ALIASES.get(folded, folded)
        return value


class BudgetComponent(BaseModel):
    """One line of work inside a budget, and the unit of chunking.

    ``description`` is what makes this component findable: ``name`` alone
    ("Authentication backend") is too short and too generic to produce a
    vector that discriminates between two different authentication jobs.
    """

    component_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8000)
    tech_stack: list[str] = Field(default_factory=list)
    estimated_hours: float = Field(gt=0, le=100_000)
    complexity: Complexity
    # Component ids this one depends on. Not embedded and not currently
    # filtered on, but kept because dropping a field at ingestion time is
    # irreversible once the source document is gone.
    dependencies: list[str] = Field(default_factory=list)

    @field_validator("component_id")
    @classmethod
    def reject_separator(cls, value: str) -> str:
        if CHUNK_ID_SEPARATOR in value:
            raise ValueError(
                f"component_id must not contain {CHUNK_ID_SEPARATOR!r}: it is the "
                f"separator in the chunk id, and a component id containing it "
                f"would make the chunk id ambiguous to parse back"
            )
        return value


class Budget(BaseModel):
    """A historical budget as it arrives from the Session 06 output."""

    budget_id: str = Field(min_length=1, max_length=64)
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1, max_length=4000)
    main_technology: str = Field(min_length=1, max_length=80)
    year: int = Field(ge=2000, le=2100)
    # Deliberately NOT validated against the sum of the components' hours.
    # A budget legitimately carries overhead that no single component owns
    # (management, contingency), and a partial component list is a normal
    # export. An equality check here would reject real data to enforce an
    # invariant the source never promised.
    total_estimated_hours: float = Field(gt=0, le=1_000_000)
    components: list[BudgetComponent] = Field(min_length=1)

    @field_validator("budget_id")
    @classmethod
    def reject_separator(cls, value: str) -> str:
        if CHUNK_ID_SEPARATOR in value:
            raise ValueError(
                f"budget_id must not contain {CHUNK_ID_SEPARATOR!r}: see "
                f"BudgetComponent.component_id for why"
            )
        return value


class Chunk(BaseModel):
    """A fragment ready to be embedded.

    The split between ``text`` and ``metadata`` is the whole design. ``text``
    is what the embedding model sees, so it carries the parent budget's
    context on purpose (see ``chunker.py``). ``metadata`` is what a vector
    store filters on *before* or *after* the similarity search: structured,
    exact, and never embedded, because "2024" as a vector is noise while
    "2024" as a filter is a hard constraint.
    """

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    # A plain dict rather than a model, unlike ``IngestStats`` below. This
    # field is a payload that travels into a vector store alongside the
    # vector, where the schema is open by design and grows one filter at a
    # time. ``chunker.py`` is the single place that builds it, so the keys
    # cannot drift even though the type does not pin them.
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    """A chunk with its vector attached."""

    embedding: list[float] = Field(min_length=1)


class IngestRequest(BaseModel):
    """Budgets to chunk and embed. They arrive in the body rather than being
    read from disk: this endpoint is a pipeline stage, not a file loader, and
    keeping it stateless is what lets the same route serve a one-off test from
    ``/docs`` and a batch from an upstream job."""

    budgets: list[Budget] = Field(min_length=1, max_length=500)


class IngestStats(BaseModel):
    """What the run cost and produced.

    Typed rather than a bare dict, because this is a fixed response
    contract: a model shows up in the OpenAPI schema with its four fields
    named, while ``dict`` shows up as an opaque object. It serialises to
    exactly the same JSON.
    """

    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    # Estimated, not billed: computed from the local tiktoken count and a
    # hardcoded price. See ``embedder.EMBEDDING_PRICE_USD_PER_MILLION_TOKENS``.
    estimated_cost_usd: float = Field(ge=0)


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
