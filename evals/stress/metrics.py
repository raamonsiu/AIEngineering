"""Stress-suite metrics: two budget contracts and one memory probe.

Why here and not in ``evals/metrics.py``
----------------------------------------
``evals/metrics.py`` holds what every suite shares: ``MetricResult``, the
``Metric`` protocol, ``run_all_metrics``. These three are different — they
are typed against observation shapes this suite invented and owns: the
``turn_observed`` event (Block 1) and the ``GET /sessions/{id}`` snapshot.
A metric that reads ``observation["latency_ms"]`` is coupled to that event,
not to ``EstimationResult``, so putting it in the shared module would make
every future suite import a dependency on a shape only this one produces.
The split is: shared *vocabulary* lives in ``evals.metrics``, shared
*subject matter* does not travel.

Budgets are contracts, not dials
--------------------------------
``LatencyBudgetMetric(budget_ms=4000)`` is the point of this suite. An SLA
written in a slide is an aspiration; the same number passed to a metric is
a test that fails. The distinction matters most for exactly the failure
mode a CAG system has — it does not crash when the context gets long, it
gets slower and dearer by degrees, and every individual turn looks fine.
A budget is what turns that gradient into a binary event with a turn number
attached.

Determinism over sophistication
-------------------------------
``MemoryDriftMetric`` is a case-insensitive substring match. No embeddings,
no LLM-as-judge. A judge model would score this suite's own subject matter
with the same non-determinism the suite exists to measure, and two runs
would differ for reasons that have nothing to do with the system under
test. The cost of the simple matcher is paraphrase blindness, which
``Fact.aliases`` mitigates by listing the surface forms a faithful system
might use (see ``evals/stress/scenarios.py``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from evals.metrics import MetricResult
from evals.stress.scenarios import DEFAULT_SLOTS, Fact


def _coerce_float(observation: Mapping[str, Any], field: str) -> tuple[float | None, str]:
    """Read a numeric field, reporting *why* rather than raising.

    A missing or unparseable field is a failed metric with an explanation,
    never an exception: see ``evals/metrics.py`` on why a long run must not
    die on one bad row.
    """
    if field not in observation:
        return None, f"observation has no '{field}' field"
    raw = observation[field]
    if raw is None:
        return None, f"'{field}' is null"
    try:
        return float(raw), ""
    except (TypeError, ValueError):
        return None, f"'{field}' is not numeric: {raw!r}"


class LatencyBudgetMetric:
    """1.0 if ``latency_ms`` <= ``budget_ms``; 0.0 otherwise.

    Reads the turn's wall-clock latency, not the sum of its LLM calls: the
    client waits for the whole turn, including attachment extraction and
    prompt rendering, and those are precisely what a large attachment makes
    expensive.
    """

    def __init__(self, budget_ms: int, *, name: str = "latency_budget") -> None:
        if budget_ms <= 0:
            raise ValueError("budget_ms must be positive")
        self.budget_ms = budget_ms
        self.name = name

    def evaluate(self, observation: Mapping[str, Any]) -> MetricResult:
        latency, problem = _coerce_float(observation, "latency_ms")
        if latency is None:
            return MetricResult(self.name, 0.0, False, problem)

        passed = latency <= self.budget_ms
        headroom = self.budget_ms - latency
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"{latency:.0f} ms vs budget {self.budget_ms} ms "
                f"({'+' if headroom >= 0 else ''}{headroom:.0f} ms headroom)"
            ),
        )


class CostBudgetMetric:
    """1.0 if ``cost_usd`` <= ``budget_usd``; 0.0 otherwise.

    ``cost_usd`` on a ``turn_observed`` row is the turn's total across every
    LLM call it made — estimator, metadata extractor and, once the window
    overflows, the summarizer. Budgeting against the estimator call alone
    would exempt the side calls from the contract at exactly the point
    where they start to matter.
    """

    def __init__(self, budget_usd: float, *, name: str = "cost_budget") -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = budget_usd
        self.name = name

    def evaluate(self, observation: Mapping[str, Any]) -> MetricResult:
        cost, problem = _coerce_float(observation, "cost_usd")
        if cost is None:
            return MetricResult(self.name, 0.0, False, problem)

        passed = cost <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"${cost:.6f} vs budget ${self.budget_usd:.6f} "
                f"({cost / self.budget_usd:.2f}x)"
            ),
        )


class MemoryDriftMetric:
    """1.0 if the tracked fact is still findable in the session's memory.

    Searches the slots named in ``where``, against a ``GET /sessions/{id}``
    snapshot:

    - ``summary``   -> the cumulative summary text
    - ``anchors``   -> the concatenated content of every anchored message
    - ``metadata``  -> the serialised ``ProjectMetadata``
    - ``response``  -> this turn's answer text, if the caller added it to
                       the snapshot as ``response_summary``. Used by the
                       attachment sweep, where the question is whether the
                       attachment's content reached the answer at all, and
                       the session's memory slots are not the right place
                       to look on a single-turn session.

    Searching all three memory slots by default is deliberate. The CAG
    design *expects* a fact to migrate — out of the sliding window, into
    the summary, or rescued into an anchor — so pinning the metric to one
    slot would score a correctly-working system as broken. ``where`` exists
    for when the question really is "did this reach the anchors", not
    "is this still known".

    A score of 1.0 means *present*, which is not the same as *correct*. For
    a fact the scenario marks ``superseded`` (an old budget, an abandoned
    stack), presence is the defect. Keeping the metric to a plain presence
    test, and leaving the sign to the scenario that declared the fact, is
    what stops it needing to understand any particular conversation.
    """

    def __init__(
        self,
        fact: str | Fact,
        where: Sequence[str] = DEFAULT_SLOTS,
        *,
        aliases: Sequence[str] = (),
        name: str | None = None,
    ) -> None:
        # ``where`` defaults to a tuple, not the list the brief sketches:
        # a mutable default argument is shared across every instance, so
        # one caller appending a slot would silently change the default for
        # every metric constructed afterwards.
        if isinstance(fact, Fact):
            self.fact_key = fact.key
            self.needles = tuple(fact.needles)
            self.expectation = fact.expectation
            self.where = tuple(where) if where is not DEFAULT_SLOTS else tuple(fact.where)
        else:
            self.fact_key = fact
            self.needles = (fact, *aliases)
            self.expectation = "persist"
            self.where = tuple(where)

        if not self.where:
            raise ValueError("where must name at least one slot")
        self.name = name or f"memory_drift_{self.fact_key}"

    @staticmethod
    def _slot_text(snapshot: Mapping[str, Any], slot: str) -> str:
        if slot == "summary":
            return snapshot.get("summary") or ""
        if slot == "anchors":
            anchors = snapshot.get("anchors") or []
            return "\n".join(
                a.get("content", "") if isinstance(a, Mapping) else str(a) for a in anchors
            )
        if slot == "metadata":
            metadata = snapshot.get("project_metadata") or {}
            # Serialised rather than walked field by field: ProjectMetadata
            # mixes scalars and lists, and a fact can land in any of them
            # (a budget may arrive as explicit_constraints text, a stack as
            # mentioned_technologies). Matching the JSON catches all of them
            # without the metric having to know the model's field layout.
            return json.dumps(metadata, ensure_ascii=False)
        if slot == "response":
            return snapshot.get("response_summary") or ""
        raise ValueError(f"unknown memory slot: {slot!r}")

    def evaluate(self, snapshot: Mapping[str, Any]) -> MetricResult:
        found_in: list[str] = []
        matched: str | None = None

        for slot in self.where:
            haystack = self._slot_text(snapshot, slot).lower()
            if not haystack:
                continue
            for needle in self.needles:
                if needle.lower() in haystack:
                    found_in.append(slot)
                    matched = matched or needle
                    break

        present = bool(found_in)
        detail = (
            f"'{matched}' found in {'+'.join(found_in)}"
            if present
            else f"none of {list(self.needles)} found in {'+'.join(self.where)}"
        )
        # ``passed`` folds in the expectation so a suite can be read at a
        # glance; ``score`` stays a raw presence signal so the CSV can be
        # re-interpreted later without re-running anything.
        passed = present if self.expectation == "persist" else not present
        return MetricResult(
            name=self.name,
            score=1.0 if present else 0.0,
            passed=passed,
            details=f"[{self.expectation}] {detail}",
        )
