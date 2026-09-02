"""Tests for the CAG-vs-RAG decision.

The point is not arithmetic — it is that the criteria stay explicit and
that the functional requirements are not overridable by headroom.
"""

from __future__ import annotations

from app.ingest.architecture import (
    PROJECT_CORPUS,
    PROJECT_MODEL,
    Architecture,
    CAGViability,
    CorpusProfile,
    ModelProfile,
    assess_cag,
    recommend_architecture,
)


def _corpus(**overrides) -> CorpusProfile:
    base = {
        "total_tokens": 10_000,
        "update_frequency_days": 180,
        "requires_source_attribution": False,
        "requires_per_user_access_control": False,
    }
    return CorpusProfile(**{**base, **overrides})


MODEL = ModelProfile(context_window=128_000, cost_per_million_input_tokens=0.15)


def test_one_failing_ceiling_is_enough_to_rule_cag_out() -> None:
    viability = CAGViability(True, True, True, quality_holds_with_load=False)

    assert viability.is_viable() is False
    assert viability.failing_constraints() == ["quality_holds_with_load"]


def test_all_four_passing_is_viable() -> None:
    assert CAGViability(True, True, True, True).is_viable() is True


def test_fitting_means_fitting_with_room() -> None:
    """Filling 95% of the window is possible and degrades quality, so
    'fits' cannot mean 'fits'."""
    tight = assess_cag(_corpus(total_tokens=120_000), MODEL)

    assert tight.fits_in_context_window is False


def test_traceability_forces_rag_regardless_of_headroom() -> None:
    """No amount of spare context window makes CAG able to cite a source."""
    tiny_but_citable = _corpus(total_tokens=500, requires_source_attribution=True)

    assert recommend_architecture(tiny_but_citable, MODEL) is Architecture.PURE_RAG


def test_per_user_access_control_forces_rag() -> None:
    tiny_but_restricted = _corpus(total_tokens=500, requires_per_user_access_control=True)

    assert recommend_architecture(tiny_but_restricted, MODEL) is Architecture.PURE_RAG


def test_a_small_and_very_stable_corpus_stays_on_cag() -> None:
    assert recommend_architecture(
        _corpus(total_tokens=5_000, update_frequency_days=365), MODEL
    ) is Architecture.PURE_CAG


def test_a_fast_moving_corpus_goes_to_rag_even_when_it_fits() -> None:
    assert recommend_architecture(
        _corpus(total_tokens=5_000, update_frequency_days=3), MODEL
    ) is Architecture.PURE_RAG


def test_the_middle_ground_is_hybrid() -> None:
    assert recommend_architecture(
        _corpus(total_tokens=50_000, update_frequency_days=30), MODEL
    ) is Architecture.HYBRID_CAG_RAG


def test_latency_is_projected_from_a_measured_fit_not_assumed() -> None:
    """Slope and intercept come from the Session 6 stress run."""
    at_zero = PROJECT_MODEL.projected_latency_ms(0)
    at_20k = PROJECT_MODEL.projected_latency_ms(20_000)

    assert at_zero == PROJECT_MODEL.latency_baseline_ms
    assert at_20k > at_zero
    # ~9.5 s at the full corpus, against a 3 s conversational budget.
    assert 9_000 < at_20k < 10_000


def test_this_project_fails_on_latency_and_quality_not_on_size() -> None:
    """The interesting result: the corpus fits and is cheap. CAG is still
    out, and naming the right reason is what makes the decision defensible."""
    viability = assess_cag(PROJECT_CORPUS, PROJECT_MODEL, quality_holds=False)

    assert viability.fits_in_context_window is True
    assert viability.cost_per_query_acceptable is True
    assert viability.failing_constraints() == [
        "latency_acceptable",
        "quality_holds_with_load",
    ]
    assert recommend_architecture(PROJECT_CORPUS, PROJECT_MODEL) is Architecture.PURE_RAG
