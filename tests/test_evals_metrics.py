"""Tests for the shared eval primitives in ``evals/metrics.py``."""

from __future__ import annotations

from evals.metrics import Metric, MetricResult, run_all_metrics


class _AlwaysPasses:
    name = "always_passes"

    def evaluate(self, observation) -> MetricResult:
        return MetricResult(name=self.name, score=1.0, passed=True, details="fine")


class _AlwaysFails:
    name = "always_fails"

    def evaluate(self, observation) -> MetricResult:
        return MetricResult(name=self.name, score=0.0, passed=False, details="nope")


class _Explodes:
    name = "explodes"

    def evaluate(self, observation) -> MetricResult:
        raise RuntimeError("boom")


def test_metric_protocol_is_structural() -> None:
    assert isinstance(_AlwaysPasses(), Metric)


def test_as_row_namespaces_columns_by_metric_name() -> None:
    row = MetricResult(name="latency_budget", score=1.0, passed=True, details="ok").as_row()

    assert row == {
        "latency_budget_score": 1.0,
        "latency_budget_passed": 1,
        "latency_budget_details": "ok",
    }


def test_run_all_metrics_preserves_declaration_order() -> None:
    results = run_all_metrics([_AlwaysFails(), _AlwaysPasses()], observation={})

    assert [r.name for r in results] == ["always_fails", "always_passes"]


def test_a_raising_metric_becomes_a_failed_result_not_a_crash() -> None:
    """One broken metric must not take down a 216-row stress run."""
    results = run_all_metrics([_Explodes(), _AlwaysPasses()], observation={})

    assert results[0].passed is False
    assert results[0].score == 0.0
    assert "RuntimeError" in results[0].details and "boom" in results[0].details
    # The metrics after it still ran.
    assert results[1].passed is True
