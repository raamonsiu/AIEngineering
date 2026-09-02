"""The CAG-vs-RAG decision, as code rather than as an opinion.

Written so the choice can be defended to someone holding a budget. The
value is not that the function is clever — it is deliberately crude —
but that it forces every criterion to be explicit and attached to a
measured number.

The four CAG ceilings operate at once, and only one has to fail:

1. **context window** — binary and obvious, and the one everybody names first.
2. **cost per query** — continuous, and usually what kills a project before
   the window does.
3. **latency** — a property of the product, not of the architecture. Fatal
   for a synchronous assistant, irrelevant for a nightly batch.
4. **attention decay over long contexts** — the most underrated. Fitting in
   the window is not the same as being processed with the fidelity of a
   short context (*lost in the middle*).

The numbers in ``PROJECT_*`` below are measured, not assumed: the corpus
figures come from this repository's own corpus, and the cost/latency
figures from the Session 6 stress run in ``evals/stress/results.csv``.
See ``CAG_LIMITS.md`` for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Architecture(Enum):
    PURE_CAG = "pure_cag"
    HYBRID_CAG_RAG = "hybrid_cag_rag"
    PURE_RAG = "pure_rag"


@dataclass(frozen=True)
class CAGViability:
    """The four ceilings. One failure is enough to rule CAG out."""

    fits_in_context_window: bool
    cost_per_query_acceptable: bool
    latency_acceptable: bool
    quality_holds_with_load: bool

    def is_viable(self) -> bool:
        return all(
            (
                self.fits_in_context_window,
                self.cost_per_query_acceptable,
                self.latency_acceptable,
                self.quality_holds_with_load,
            )
        )

    def failing_constraints(self) -> list[str]:
        return [name for name, ok in vars(self).items() if not ok]


@dataclass(frozen=True)
class CorpusProfile:
    total_tokens: int
    update_frequency_days: int
    requires_source_attribution: bool
    requires_per_user_access_control: bool


@dataclass(frozen=True)
class ModelProfile:
    context_window: int
    cost_per_million_input_tokens: float
    # Measured, not guessed: slope and intercept of latency against input
    # tokens, fitted on the attachment sweep of the Session 6 stress run
    # (the one sweep with no compression confound). See CAG_LIMITS.md.
    latency_ms_per_token: float = 0.222
    latency_baseline_ms: float = 5078.0

    def projected_latency_ms(self, tokens: int) -> float:
        return self.latency_baseline_ms + self.latency_ms_per_token * tokens

    def projected_cost_usd(self, tokens: int) -> float:
        return tokens / 1_000_000 * self.cost_per_million_input_tokens


def assess_cag(
    corpus: CorpusProfile,
    model: ModelProfile,
    *,
    max_context_usage: float = 0.70,
    latency_budget_ms: float = 3_000.0,
    cost_budget_usd: float = 0.05,
    quality_holds: bool = True,
) -> CAGViability:
    """Turn measurements into the four booleans.

    ``max_context_usage`` defaults to 0.70 rather than 1.0 on purpose:
    filling 95% of the window is technically possible and degrades
    quality, so "fits" has to mean "fits with room", not "fits".

    ``quality_holds`` is passed in rather than computed. It is the one
    ceiling no formula can establish — it needs a measured recall curve
    over a loaded context, which is exactly what the Session 6 stress run
    produced. Pretending to derive it here would be the kind of invented
    number this module exists to avoid.
    """
    return CAGViability(
        fits_in_context_window=(corpus.total_tokens / model.context_window) <= max_context_usage,
        cost_per_query_acceptable=model.projected_cost_usd(corpus.total_tokens) <= cost_budget_usd,
        latency_acceptable=model.projected_latency_ms(corpus.total_tokens) <= latency_budget_ms,
        quality_holds_with_load=quality_holds,
    )


def recommend_architecture(corpus: CorpusProfile, model: ModelProfile) -> Architecture:
    """The decision tree. Order matters: the functional requirements are
    checked before the quantitative ones, because no amount of headroom in
    the context window makes CAG able to cite a source."""
    # Traceability is not negotiable by size. CAG injects the whole corpus
    # and cannot attribute a claim to a specific fragment.
    if corpus.requires_source_attribution:
        return Architecture.PURE_RAG

    # Per-user access control is equally structural: everything in the
    # prompt is visible to whoever asked.
    if corpus.requires_per_user_access_control:
        return Architecture.PURE_RAG

    context_usage = corpus.total_tokens / model.context_window
    if context_usage > 0.70:
        return Architecture.PURE_RAG
    if corpus.update_frequency_days < 7:
        return Architecture.PURE_RAG
    if corpus.update_frequency_days > 90 and context_usage < 0.30:
        return Architecture.PURE_CAG

    return Architecture.HYBRID_CAG_RAG


# ----------------------------------------------------------------------
# This project's measured profile
# ----------------------------------------------------------------------
# Corpus: 79,954 characters of extractable text across all five catalog
# sources (~19,988 tokens at ~4 chars/token). Measured by running every
# parser over data/corpus/.
PROJECT_CORPUS = CorpusProfile(
    total_tokens=19_988,
    # New client meetings weekly; closed budgets monthly. The binding
    # cadence is the fastest-moving source that matters.
    update_frequency_days=7,
    # A 80,000 EUR estimate the sales team cannot trace to precedents is
    # an estimate they cannot defend in front of the client.
    requires_source_attribution=True,
    # Transcripts and contracts are confidential per client and per team.
    requires_per_user_access_control=True,
)

PROJECT_MODEL = ModelProfile(
    context_window=128_000,
    cost_per_million_input_tokens=0.15,  # gpt-4o-mini, app/constants.py
)
