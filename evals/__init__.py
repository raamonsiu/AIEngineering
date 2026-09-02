"""Evaluation framework for the CAG estimator.

``evals.metrics`` holds the primitives every suite shares (``MetricResult``,
the ``Metric`` protocol, ``run_all_metrics``). Suites live in subpackages:
``evals.stress`` is the load/degradation suite (Session 6).
"""

from evals.metrics import Metric, MetricResult, run_all_metrics

__all__ = ["Metric", "MetricResult", "run_all_metrics"]
