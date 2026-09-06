"""Structural chunking: the JSON already knows where the seams are.

A budget is not prose. Somebody has already divided it into components,
each with its own scope, stack and estimate, and that division carries
more meaning than any character count could recover. So the strategy here
is to not split at all — one component is one chunk — and to let the
document's own structure decide the boundaries.

Two consequences follow, and both are deliberate:

- **No overlap.** Overlap exists to repair boundaries that a blind
  splitter chose badly. There are no blind boundaries here, so the
  repair has nothing to repair and would only duplicate tokens.
- **No fixed-size splitting of long descriptions.** A component whose
  description does not fit is a *finding*, not a case to handle: it means
  the upstream budget is using one component to describe several. The
  chunker logs it (see ``MAX_TOKENS_PER_CHUNK``) and passes it through
  rather than hiding it behind a splitter.

The one thing the chunker does add is the parent context header. Without
it the vector for "OAuth 2.0 authentication backend" is identical whether
the work was quoted for a bank or for a bike shop, and every retrieval
that needs "what did we charge *fintech* clients for auth" fails. This is
the contextual-chunk-header idea: a few tokens of parent context prepended
to each child so the fragment stays interpretable on its own.
"""

from __future__ import annotations

import structlog
import tiktoken

from app.embedding_pipeline.schemas import (
    CHUNK_ID_SEPARATOR,
    Budget,
    BudgetComponent,
    Chunk,
)

log = structlog.get_logger()

# The model the counts have to be right for. Chunking against one
# tokenizer and embedding with another would make ``token_count`` a
# decorative number, so this is pinned to the embedder's model on purpose.
TOKENIZER_MODEL = "text-embedding-3-small"

# text-embedding-3-small's context limit. A chunk above it is rejected by
# the API, not truncated, so the whole batch would fail. Checked here, one
# stage before the network call, because that is where the chunk id is
# still available to say *which* component is at fault.
MAX_TOKENS_PER_CHUNK = 8191

# The filterable fields every chunk is contractually required to carry.
# A vector store query that filters on a key missing from some chunks
# silently returns fewer results instead of erroring, so this set is
# asserted in the tests rather than trusted.
REQUIRED_METADATA_KEYS = frozenset(
    {
        "budget_id",
        "component_id",
        "client_sector",
        "main_technology",
        "year",
        "complexity",
        "estimated_hours",
    }
)


def _format_hours(hours: float) -> str:
    """Render 120.0 as "120" and 7.5 as "7.5".

    Cosmetic, but it lands in the embedded text: a trailing ".0" on every
    component is a token spent on nothing.
    """
    return str(int(hours)) if hours == int(hours) else str(hours)


class JSONStructuralChunker:
    """One budget component in, one chunk out."""

    def __init__(self, tokenizer_model: str = TOKENIZER_MODEL) -> None:
        # Built once and reused. ``encoding_for_model`` resolves and caches
        # a BPE file on first use; doing it per component would pay that
        # lookup sixty-seven times for one request.
        self._encoding = tiktoken.encoding_for_model(tokenizer_model)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        """Flatten budgets into chunks, preserving document order.

        Order matters downstream: the embedder zips its API response back
        onto this list positionally, so the sequence is part of the
        contract between the two stages.
        """
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                chunks.append(self._build_chunk(budget, component))

        oversized = [c for c in chunks if c.token_count > MAX_TOKENS_PER_CHUNK]
        if oversized:
            # Reported, not fixed. Splitting here would make the budget
            # look well-formed while hiding that one component is doing
            # the job of three.
            log.warning(
                "chunks_exceed_model_limit",
                limit=MAX_TOKENS_PER_CHUNK,
                count=len(oversized),
                chunk_ids=[c.chunk_id for c in oversized][:10],
            )

        log.info(
            "chunking_completed",
            budgets=len(budgets),
            chunks=len(chunks),
            tokens=sum(c.token_count for c in chunks),
            max_chunk_tokens=max((c.token_count for c in chunks), default=0),
        )
        return chunks

    def _build_chunk(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self._render(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}{CHUNK_ID_SEPARATOR}{component.component_id}",
            text=text,
            metadata=self._build_metadata(budget, component),
            token_count=len(self._encoding.encode(text)),
        )

    def _render(self, budget: Budget, component: BudgetComponent) -> str:
        """The text that actually gets embedded.

        The two bracketed header lines are the parent context; everything
        below is the component itself. Labels ("Component:", "Tech stack:")
        are kept because they give the model something to anchor on when a
        field is empty — a bare empty line is ambiguous, "Tech stack: " is
        explicitly nothing.
        """
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | "
            f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
            f"\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {_format_hours(component.estimated_hours)}"
        )

    def _build_metadata(self, budget: Budget, component: BudgetComponent) -> dict:
        """Filterable fields, kept out of the embedded text on purpose.

        ``year`` and ``estimated_hours`` are the clearest case: as part of
        the text they are a handful of tokens that blur the vector, while
        as metadata they support "components from 2024 under 100 hours",
        which similarity search cannot express at all.
        """
        return {
            # The seven the contract requires.
            "budget_id": budget.budget_id,
            "component_id": component.component_id,
            "client_sector": budget.client_metadata.sector,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "complexity": component.complexity,
            "estimated_hours": component.estimated_hours,
            # Two more the corpus makes obviously useful: "what did we
            # quote this client before" and "which market was this for".
            "client_name": budget.client_metadata.name,
            "client_country": budget.client_metadata.country,
        }
