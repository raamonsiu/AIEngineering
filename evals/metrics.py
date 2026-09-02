"""The shared vocabulary every eval in this repo speaks: ``MetricResult``.

A metric is anything with a ``name`` and an ``evaluate(observation)`` that
returns a ``MetricResult``. That is the whole contract — deliberately so.

Design notes
------------
- ``score`` is a float and ``passed`` a bool, kept as *separate* fields
  rather than deriving one from the other. A binary metric sets
  ``score = 1.0 if passed else 0.0``, but a future graded metric (recall at
  0.72) still needs a threshold to say whether that counts as a pass, and
  the threshold belongs to the metric, not to the reader of the result.
  Averaging ``score`` across runs and averaging ``passed`` then mean two
  different, both-useful things.
- ``details`` is free text aimed at a human reading a failure, not a
  machine. It should say what was observed versus what was expected
  ("latency 5210 ms > budget 4000 ms"), because a metric that only reports
  0.0 makes you re-run the whole suite to find out why.
- Metrics never raise on a missing field: an observation that lacks the
  field a metric reads is a *failed* metric with an explanatory
  ``details``, not a crashed run. A stress run that dies on turn 140 of 216
  loses the whole dataset, which is a worse outcome than a row of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MetricResult:
    """The outcome of evaluating one metric against one observation."""

    name: str
    score: float
    passed: bool
    details: str = ""

    def as_row(self) -> dict[str, Any]:
        """Flatten to CSV-friendly columns, namespaced by metric name, so
        several metrics can be merged into a single row without colliding."""
        return {
            f"{self.name}_score": self.score,
            f"{self.name}_passed": int(self.passed),
            f"{self.name}_details": self.details,
        }


@runtime_checkable
class Metric(Protocol):
    """Structural type: no inheritance required, just the two members.

    Keeping this a ``Protocol`` rather than an ABC means a metric can be a
    dataclass, a closure-backed object or a plain class without being
    coupled to this module's import graph.
    """

    name: str

    def evaluate(self, observation: Any) -> MetricResult: ...


def run_all_metrics(metrics: list[Metric], observation: Any) -> list[MetricResult]:
    """Evaluate every metric against one observation, in declaration order.

    A metric that raises is converted into a failed ``MetricResult`` rather
    than propagating: one broken metric must not take down a suite, and the
    traceback's type/message is preserved in ``details`` so the failure is
    still diagnosable from the CSV alone.
    """
    results: list[MetricResult] = []
    for metric in metrics:
        try:
            results.append(metric.evaluate(observation))
        except Exception as exc:  # noqa: BLE001 — see docstring
            results.append(
                MetricResult(
                    name=getattr(metric, "name", type(metric).__name__),
                    score=0.0,
                    passed=False,
                    details=f"metric raised {type(exc).__name__}: {exc}",
                )
            )
    return results
