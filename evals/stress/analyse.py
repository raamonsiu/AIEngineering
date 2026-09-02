"""Turn ``results.csv`` into the tables that go in ``REPORT.md``.

    uv run python -m evals.stress.analyse --input evals/stress/results.csv

A script rather than hand-typed numbers: every figure in the report is then
reproducible from the CSV, and a re-run cannot leave the prose quietly
disagreeing with the data.

Percentiles use the nearest-rank method, not interpolation. With nine
samples per cell, an interpolated P95 is a number no run actually produced;
nearest-rank returns an observation that really happened, which is what you
want when the next question is "show me that turn".
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

TURN_LADDER = (1, 3, 6, 10, 20)


def _num(row: dict, key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: always returns an observed value."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def _fmt(value: float, spec: str = ".0f") -> str:
    return "—" if value != value else format(value, spec)  # NaN check


def load(path: Path) -> tuple[list[dict], list[dict]]:
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))]
    ok = [r for r in rows if not (r.get("error") or "").strip()]
    return rows, ok


# ----------------------------------------------------------------------
def table_summary(rows: list[dict]) -> str:
    """One line per scenario x attachment size, as the brief asks."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["sweep"], r["scenario"], r["attachment_size_kb"])].append(r)

    out = [
        "| sweep | scenario | attach KB | n | P50 lat (ms) | P95 lat (ms) | total cost (USD) "
        "| exact hit | semantic hit | mean recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (sweep, scenario, kb), group in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][2]))
    ):
        lat = [v for v in (_num(r, "latency_ms") for r in group) if v is not None]
        cost = [v for v in (_num(r, "cost_usd") for r in group) if v is not None]
        recall = [v for v in (_num(r, "memory_drift_mean") for r in group) if v is not None]
        kinds = [r.get("cache_hit_kind", "") for r in group]
        exact = sum(1 for k in kinds if k == "exact") / len(kinds) if kinds else 0.0
        semantic = sum(1 for k in kinds if k == "semantic") / len(kinds) if kinds else 0.0
        out.append(
            f"| {sweep} | {scenario} | {kb} | {len(group)} | "
            f"{_fmt(percentile(lat, 50))} | {_fmt(percentile(lat, 95))} | "
            f"{_fmt(sum(cost), '.4f')} | {exact:.0%} | {semantic:.0%} | "
            f"{_fmt(_mean(recall), '.3f')} |"
        )
    return "\n".join(out)


def curve_latency_vs_tokens(rows: list[dict]) -> str:
    """Curve 1. Bucketed by tokens_in so the relationship is visible
    without a scatter plot."""
    edges = [(0, 3500), (3500, 4500), (4500, 5500), (5500, 6500), (6500, 9000), (9000, 10**9)]
    out = [
        "| tokens_in bucket | n | P50 lat (ms) | P95 lat (ms) | mean cost (USD) | mean LLM calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for low, high in edges:
        group = [
            r
            for r in rows
            if (t := _num(r, "tokens_in")) is not None and low <= t < high
        ]
        if not group:
            continue
        lat = [_num(r, "latency_ms") or 0 for r in group]
        label = f"{low:,}–{high:,}" if high < 10**9 else f"{low:,}+"
        out.append(
            f"| {label} | {len(group)} | {_fmt(percentile(lat, 50))} | "
            f"{_fmt(percentile(lat, 95))} | "
            f"{_fmt(_mean(_num(r, 'cost_usd') or 0 for r in group), '.6f')} | "
            f"{_fmt(_mean(_num(r, 'llm_calls') or 0 for r in group), '.2f')} |"
        )
    return "\n".join(out)


def curve_cost_vs_turn(rows: list[dict]) -> str:
    """Curve 2: cumulative cost per scenario at each rung of the ladder."""
    turn_rows = [r for r in rows if r["sweep"] == "turns"]
    scenarios = sorted({r["scenario"] for r in turn_rows})

    out = [
        "| turn | " + " | ".join(f"{s} cum USD" for s in scenarios) + " | mean turn USD | mean lat (ms) |",
        "|---:|" + "---:|" * (len(scenarios) + 2),
    ]
    for n in TURN_LADDER:
        at_n = [r for r in turn_rows if int(r["turn_index"]) == n]
        if not at_n:
            continue
        cells = []
        for s in scenarios:
            vals = [_num(r, "cum_cost_usd") for r in at_n if r["scenario"] == s]
            cells.append(_fmt(_mean(v for v in vals if v is not None), ".5f"))
        out.append(
            f"| {n} | " + " | ".join(cells) + " | "
            f"{_fmt(_mean(_num(r, 'cost_usd') or 0 for r in at_n), '.6f')} | "
            f"{_fmt(_mean(_num(r, 'latency_ms') or 0 for r in at_n))} |"
        )
    return "\n".join(out)


def curve_recall_vs_turn(rows: list[dict]) -> str:
    """Curve 3: MemoryDriftMetric against conversation length."""
    turn_rows = [r for r in rows if r["sweep"] == "turns"]
    scenarios = sorted({r["scenario"] for r in turn_rows})

    out = [
        "| turn | " + " | ".join(f"{s} recall" for s in scenarios)
        + " | project_name recall | stale facts still present |",
        "|---:|" + "---:|" * (len(scenarios) + 2),
    ]
    for n in TURN_LADDER:
        at_n = [r for r in turn_rows if int(r["turn_index"]) == n]
        if not at_n:
            continue
        cells = []
        for s in scenarios:
            vals = [_num(r, "memory_drift_mean") for r in at_n if r["scenario"] == s]
            cells.append(_fmt(_mean(v for v in vals if v is not None), ".3f"))
        names = [v for r in at_n if (v := _num(r, "memory_drift_project_name_score")) is not None]
        stale_total = sum(_num(r, "stale_facts_total") or 0 for r in at_n)
        stale_present = sum(_num(r, "stale_facts_present") or 0 for r in at_n)
        out.append(
            f"| {n} | " + " | ".join(cells) + " | "
            f"{_fmt(_mean(names), '.3f')} | "
            f"{stale_present:.0f}/{stale_total:.0f} |"
        )
    return "\n".join(out)


def table_attachments(rows: list[dict]) -> str:
    """The attachment sweep, aggregated across scenarios."""
    att = [r for r in rows if r["sweep"] == "attachments"]
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in att:
        by_size[int(r["attachment_size_kb"])].append(r)

    out = [
        "| attach KB | n | extracted chars | accepted | P50 lat (ms) | P95 lat (ms) "
        "| mean tokens_in | mean cost (USD) | marker recall (head/mid/tail) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for kb in sorted(by_size):
        group = by_size[kb]
        lat = [_num(r, "latency_ms") or 0 for r in group]
        accepted = sum(1 for r in group if (_num(r, "attachments_rejected") or 0) == 0)
        positions = {"head": [], "middle": [], "tail": []}
        for r in group:
            for part in (r.get("attachment_recall_detail") or "").split(";"):
                if "=" in part:
                    pos, val = part.split("=", 1)
                    if pos in positions:
                        positions[pos].append(float(val))
        marker = (
            " / ".join(
                _fmt(_mean(positions[p]), ".2f") if positions[p] else "—"
                for p in ("head", "middle", "tail")
            )
            if kb
            else "n/a"
        )
        out.append(
            f"| {kb} | {len(group)} | "
            f"{_fmt(_mean(_num(r, 'attachments_total_chars') or 0 for r in group), ',.0f')} | "
            f"{accepted}/{len(group)} | {_fmt(percentile(lat, 50))} | {_fmt(percentile(lat, 95))} | "
            f"{_fmt(_mean(_num(r, 'tokens_in') or 0 for r in group), ',.0f')} | "
            f"{_fmt(_mean(_num(r, 'cost_usd') or 0 for r in group), '.6f')} | {marker} |"
        )
    return "\n".join(out)


def table_budget_crossings(rows: list[dict], thresholds=(4000, 6000, 8000, 10000, 15000)) -> str:
    """At which turn does each candidate latency SLA start failing?

    Computed from the raw ``latency_ms`` column rather than from the metric
    column, so the report can discuss contracts other than the one the run
    was configured with without re-running anything — which is the point of
    keeping raw observations next to derived verdicts.
    """
    turn_rows = [r for r in rows if r["sweep"] == "turns"]
    out = [
        "| latency SLA | turns passing | first turn where P50 breaches |",
        "|---:|---:|---:|",
    ]
    by_turn: dict[int, list[float]] = defaultdict(list)
    for r in turn_rows:
        if (v := _num(r, "latency_ms")) is not None:
            by_turn[int(r["turn_index"])].append(v)

    for limit in thresholds:
        passing = sum(
            1 for r in turn_rows if (v := _num(r, "latency_ms")) is not None and v <= limit
        )
        total = sum(1 for r in turn_rows if _num(r, "latency_ms") is not None)
        first = next(
            (n for n in sorted(by_turn) if percentile(by_turn[n], 50) > limit), None
        )
        out.append(
            f"| {limit:,} ms | {passing}/{total} ({passing / total:.0%}) | "
            f"{first if first is not None else 'never'} |"
        )
    return "\n".join(out)


def key_facts(rows: list[dict]) -> str:
    """The concrete quantitative claims the report's prose needs."""
    turn_rows = [r for r in rows if r["sweep"] == "turns"]
    lines: list[str] = []

    def at(n: int, field: str) -> float:
        return _mean(
            v
            for r in turn_rows
            if int(r["turn_index"]) == n and (v := _num(r, field)) is not None
        )

    for field, spec in (("latency_ms", ".0f"), ("cost_usd", ".6f"), ("tokens_in", ".0f")):
        t1, t20 = at(1, field), at(20, field)
        # NaN != NaN; a partial run has no turn 20 yet and must not print
        # "nanx" as if it were a measured ratio.
        ratio = f" ({t20 / t1:.2f}x)" if t1 == t1 and t20 == t20 and t1 else ""
        lines.append(
            f"- mean {field}: turn 1 = {_fmt(t1, spec)}, turn 20 = {_fmt(t20, spec)}{ratio}"
        )

    summaries = [_num(r, "summary_chars") or 0 for r in turn_rows]
    lines.append(
        f"- summary_chars: max observed = {max(summaries):.0f}, "
        f"non-zero on {sum(1 for s in summaries if s > 0)}/{len(summaries)} turns"
    )
    anchors = [_num(r, "anchors_count") or 0 for r in turn_rows]
    lines.append(f"- anchors_count: max observed = {max(anchors):.0f}")
    kinds = [r.get("cache_hit_kind") for r in rows]
    lines.append(
        f"- cache_hit_kind: "
        + ", ".join(f"{k}={kinds.count(k)}" for k in sorted(set(kinds)))
        + f" of {len(kinds)} rows"
    )
    calls = defaultdict(list)
    for r in turn_rows:
        calls[int(r["turn_index"])].append(_num(r, "llm_calls") or 0)
    lines.append(
        "- mean llm_calls by turn: "
        + ", ".join(f"t{n}={_mean(calls[n]):.2f}" for n in TURN_LADDER if n in calls)
    )

    # Where does project_name recall first drop below 60%?
    by_turn = defaultdict(list)
    for r in turn_rows:
        if (v := _num(r, "memory_drift_project_name_score")) is not None:
            by_turn[int(r["turn_index"])].append(v)
    below = next((n for n in sorted(by_turn) if _mean(by_turn[n]) < 0.6), None)
    lines.append(
        f"- project_name recall first below 60% at turn {below}"
        if below
        else "- project_name recall never drops below 60%"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="evals/stress/results.csv")
    args = parser.parse_args()

    all_rows, rows = load(Path(args.input))
    failed = len(all_rows) - len(rows)
    print(f"# Derived tables\n\n{len(all_rows)} rows ({failed} failed, {len(rows)} scored)\n")
    for title, body in (
        ("Summary", table_summary(rows)),
        ("Curve 1 — latency vs tokens_in", curve_latency_vs_tokens(rows)),
        ("Curve 2 — cumulative cost vs turn", curve_cost_vs_turn(rows)),
        ("Curve 3 — recall vs N", curve_recall_vs_turn(rows)),
        ("Attachment sweep", table_attachments(rows)),
        ("Latency SLA crossings", table_budget_crossings(rows)),
        ("Key facts", key_facts(rows)),
    ):
        print(f"\n## {title}\n\n{body}")


if __name__ == "__main__":
    main()
