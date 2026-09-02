"""Unit tests for the stress-suite metrics (Session 6, Block 4).

Each metric gets a passing case, a failing case and a boundary/degenerate
case, because the boundary is where a budget metric is actually used: the
interesting question is never "is 50 ms under 4000 ms", it is "is exactly
4000 ms a pass".
"""

from __future__ import annotations

import pytest

from evals.metrics import run_all_metrics
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import Fact


# ----------------------------------------------------------------------
# LatencyBudgetMetric
# ----------------------------------------------------------------------
def test_latency_under_budget_passes() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 2500})

    assert result.passed is True
    assert result.score == 1.0
    assert "2500 ms" in result.details


def test_latency_over_budget_fails() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 5210})

    assert result.passed is False
    assert result.score == 0.0
    assert "-1210 ms headroom" in result.details


def test_latency_exactly_at_budget_passes() -> None:
    """The budget is a ceiling the system is allowed to touch, not to cross."""
    assert LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4000}).passed is True
    assert LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4001}).passed is False


def test_latency_with_a_missing_field_fails_without_raising() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"cost_usd": 0.001})

    assert result.passed is False
    assert "no 'latency_ms' field" in result.details


def test_a_nonsensical_budget_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetMetric(budget_ms=0)


# ----------------------------------------------------------------------
# CostBudgetMetric
# ----------------------------------------------------------------------
def test_cost_under_budget_passes() -> None:
    result = CostBudgetMetric(budget_usd=0.005).evaluate({"cost_usd": 0.0012})

    assert result.passed is True
    assert result.score == 1.0
    assert "0.24x" in result.details


def test_cost_over_budget_fails() -> None:
    result = CostBudgetMetric(budget_usd=0.005).evaluate({"cost_usd": 0.0125})

    assert result.passed is False
    assert "2.50x" in result.details


def test_cost_exactly_at_budget_passes() -> None:
    assert CostBudgetMetric(budget_usd=0.005).evaluate({"cost_usd": 0.005}).passed is True


def test_a_free_turn_passes_any_budget() -> None:
    """A cache hit spends nothing; it must not read as a budget violation."""
    assert CostBudgetMetric(budget_usd=0.0001).evaluate({"cost_usd": 0.0}).passed is True


# ----------------------------------------------------------------------
# MemoryDriftMetric
# ----------------------------------------------------------------------
_EMPTY_SNAPSHOT = {"summary": None, "anchors": [], "project_metadata": {}}


def test_fact_present_in_the_summary_is_found() -> None:
    snapshot = {**_EMPTY_SNAPSHOT, "summary": "The Nimbus platform needs SAML SSO."}

    result = MemoryDriftMetric("Nimbus").evaluate(snapshot)

    assert result.passed is True
    assert result.score == 1.0
    assert "found in summary" in result.details


def test_fact_absent_everywhere_is_drift() -> None:
    snapshot = {**_EMPTY_SNAPSHOT, "summary": "A platform for scheduling maintenance."}

    result = MemoryDriftMetric("Nimbus").evaluate(snapshot)

    assert result.passed is False
    assert result.score == 0.0
    assert "none of ['Nimbus']" in result.details


def test_matching_is_case_insensitive() -> None:
    snapshot = {**_EMPTY_SNAPSHOT, "summary": "the NIMBUS rollout"}

    assert MemoryDriftMetric("Nimbus").evaluate(snapshot).score == 1.0


def test_fact_found_in_anchors() -> None:
    snapshot = {
        **_EMPTY_SNAPSHOT,
        "anchors": [{"role": "user", "content": "The budget is locked at 30000 EUR."}],
    }

    result = MemoryDriftMetric("30000").evaluate(snapshot)

    assert result.score == 1.0
    assert "found in anchors" in result.details


def test_fact_found_inside_serialised_metadata_lists() -> None:
    """A fact can land in any ProjectMetadata field; the metric must not
    need to know which one."""
    snapshot = {
        **_EMPTY_SNAPSHOT,
        "project_metadata": {
            "project_name": None,
            "mentioned_technologies": ["Flutter", "Dart"],
            "explicit_constraints": [],
        },
    }

    assert MemoryDriftMetric("Flutter").evaluate(snapshot).score == 1.0


def test_aliases_stop_the_metric_measuring_formatting() -> None:
    """A summarizer writing '30,000 EUR' has not forgotten the budget."""
    snapshot = {**_EMPTY_SNAPSHOT, "summary": "Budget ceiling of 30,000 EUR agreed."}

    bare = MemoryDriftMetric("30000")
    with_aliases = MemoryDriftMetric("30000", aliases=("30,000", "30.000"))

    assert bare.evaluate(snapshot).score == 0.0
    assert with_aliases.evaluate(snapshot).score == 1.0


def test_where_restricts_the_slots_searched() -> None:
    snapshot = {**_EMPTY_SNAPSHOT, "summary": "The Nimbus platform."}

    assert MemoryDriftMetric("Nimbus", where=["summary"]).evaluate(snapshot).score == 1.0
    assert MemoryDriftMetric("Nimbus", where=["anchors"]).evaluate(snapshot).score == 0.0


def test_an_unknown_slot_is_rejected_rather_than_silently_skipped() -> None:
    with pytest.raises(ValueError):
        MemoryDriftMetric("Nimbus", where=[]).evaluate(_EMPTY_SNAPSHOT)
    with pytest.raises(ValueError):
        MemoryDriftMetric("Nimbus", where=["nonsense"]).evaluate(_EMPTY_SNAPSHOT)


def test_a_superseded_fact_inverts_passed_but_not_score() -> None:
    """A stale value still present is a defect, and the raw presence signal
    stays in ``score`` so the CSV can be re-read without re-running."""
    stale = Fact(
        key="budget_superseded",
        value="30000",
        introduced_at_turn=3,
        expectation="superseded",
        superseded_at_turn=8,
    )
    still_there = {**_EMPTY_SNAPSHOT, "summary": "Budget locked at 30000 EUR."}
    gone = {**_EMPTY_SNAPSHOT, "summary": "Budget locked at 80000 EUR."}

    assert MemoryDriftMetric(stale).evaluate(still_there).score == 1.0
    assert MemoryDriftMetric(stale).evaluate(still_there).passed is False
    assert MemoryDriftMetric(stale).evaluate(gone).score == 0.0
    assert MemoryDriftMetric(stale).evaluate(gone).passed is True


def test_a_fact_object_supplies_its_own_aliases_and_name() -> None:
    fact = Fact(key="budget_current", value="80000", introduced_at_turn=8, aliases=("80,000",))
    snapshot = {**_EMPTY_SNAPSHOT, "summary": "Ceiling raised to 80,000 EUR."}

    result = MemoryDriftMetric(fact).evaluate(snapshot)

    assert result.name == "memory_drift_budget_current"
    assert result.passed is True


def test_the_response_slot_covers_attachment_recall() -> None:
    """Single-turn attachment runs ask a different question: did the
    attachment's content reach the answer at all."""
    snapshot = {
        **_EMPTY_SNAPSHOT,
        "response_summary": "Estimate for engagement ORION-7731, pilot at the Zaragoza depot.",
    }

    assert MemoryDriftMetric("ORION-7731", where=["response"]).evaluate(snapshot).score == 1.0
    assert MemoryDriftMetric("14 March 2027", where=["response"]).evaluate(snapshot).score == 0.0


# ----------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------
def test_metrics_compose_into_namespaced_csv_columns() -> None:
    observation = {"latency_ms": 5210, "cost_usd": 0.0012}
    results = run_all_metrics(
        [LatencyBudgetMetric(4000), CostBudgetMetric(0.005)], observation
    )

    row: dict = {}
    for result in results:
        row.update(result.as_row())

    assert row["latency_budget_passed"] == 0
    assert row["cost_budget_passed"] == 1
