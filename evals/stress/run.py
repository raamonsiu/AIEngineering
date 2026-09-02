"""Stress runner: drives the CAG under load and writes one CSV row per turn.

    uv run python -m evals.stress.run --http http://localhost:8000 \
        --scenarios growing,pivot,contradiction \
        --attachment-sizes 0,5,20,50,100 \
        --repeats 3 \
        --output evals/stress/results.csv

Two sweeps, not one cross-product
---------------------------------
The brief sketches a triple-nested loop over scenarios x attachment sizes x
repeats. Run literally over 20-turn scenarios that is 900 turns and ~2,700
LLM calls, and — worse than the cost — it confounds the two things being
measured. Block 3 states the attachment stress explicitly: *the same
initial estimate, same short transcript, only the attachment varies*. A
20-turn conversation that also carries a 50 KB PDF on every turn cannot
tell you whether turn 14 was slow because of the history or because of the
attachment.

So the runner performs two sweeps that vary one thing each:

- ``turns``       — scenario x repeat, no attachment, turns 1..N. Isolates
                    conversation length.
- ``attachments`` — scenario x size x repeat, exactly one turn. Isolates
                    attachment size.

The ladder N in {1, 3, 6, 10, 20} is read off the turn sweep's rows rather
than run as five separate sessions: a 20-turn session *contains* its own
1-, 3-, 6- and 10-turn prefixes, and re-running them as independent
sessions would spend four times the budget to regenerate data the long run
already produced — while introducing variance between the prefix and the
run it is supposed to be a prefix of.

Reading ``turn_observed``
-------------------------
Over HTTP rather than by scraping the estimator's stdout. Parsing
``docker compose logs | grep turn_observed`` means reconstructing which
line belongs to which request, re-parsing structlog's console renderer
(which is not a stable format), and racing the log flush against the next
request. ``GET /sessions/{id}`` returns the same record as structured JSON,
already attached to the session that produced it. The event is still
logged — the log is for tracing, the endpoint is for datasets.

Rows are flushed as they are produced, so a run interrupted at turn 140 of
225 still leaves 140 usable rows rather than nothing.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from evals.metrics import run_all_metrics
from evals.stress.fixtures.build_pdfs import MARKERS, ensure_corpus
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import Fact, Scenario, get_scenarios

API = "/api/v1"

COLUMNS = [
    # --- run coordinates ---------------------------------------------
    "sweep",
    "scenario",
    "repeat",
    "attachment_size_kb",
    # --- turn_observed, verbatim --------------------------------------
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "llm_calls",
    "llm_latency_ms",
    "model",
    "attachments_count",
    "attachments_rejected",
    # --- derived -------------------------------------------------------
    "cum_cost_usd",
    "confidence_pct",
    "total_cost_eur",
    # --- binary metrics -------------------------------------------------
    "latency_budget_score",
    "latency_budget_passed",
    "cost_budget_score",
    "cost_budget_passed",
    # --- memory drift ----------------------------------------------------
    "memory_drift_mean",
    "memory_drift_project_name_score",
    "persist_facts_total",
    "persist_facts_recalled",
    "stale_facts_total",
    "stale_facts_present",
    "memory_drift_detail",
    # --- attachment recall -----------------------------------------------
    "attachment_recall_mean",
    "attachment_recall_detail",
    # --- failure bookkeeping ----------------------------------------------
    "error",
]


# ----------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------
@dataclass
class Client:
    """Thin adapter over httpx (``--http``) or FastAPI's TestClient.

    In-process is the default because it needs no running service, no Redis
    and no container; ``--http`` is what the brief asks for and is the
    honest mode for latency numbers, since it includes serialisation and
    the network hop a real client pays.
    """

    inner: Any
    base_url: str = ""

    @classmethod
    def over_http(cls, base_url: str, timeout: float) -> "Client":
        import httpx

        return cls(inner=httpx.Client(timeout=timeout), base_url=base_url.rstrip("/"))

    @classmethod
    def in_process(cls) -> "Client":
        from fastapi.testclient import TestClient

        from app.main import app

        return cls(inner=TestClient(app), base_url="")

    def create_session(self) -> str:
        response = self.inner.post(f"{self.base_url}{API}/sessions")
        response.raise_for_status()
        return response.json()["session_id"]

    def estimate(self, session_id: str, transcript: str, pdf: Path | None) -> dict:
        files = None
        if pdf is not None:
            files = {"attachments": (pdf.name, pdf.read_bytes(), "application/pdf")}
        response = self.inner.post(
            f"{self.base_url}{API}/sessions/{session_id}/estimate",
            data={"transcript": transcript},
            files=files,
        )
        response.raise_for_status()
        return response.json()

    def snapshot(self, session_id: str) -> dict:
        response = self.inner.get(f"{self.base_url}{API}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()


# ----------------------------------------------------------------------
# Scoring one turn
# ----------------------------------------------------------------------
def _response_text(result: dict) -> str:
    """Everything the answer said, not just its headline summary.

    The brief asks whether the attachment's content reached "the summary of
    the response". Taking that literally would score a model that mentions
    the pilot site inside a phase description as having forgotten it, which
    is not what recall is supposed to mean, so the phases' own text is
    folded in.
    """
    parts = [result.get("summary", "")]
    for phase in result.get("phases", []) or []:
        parts.append(phase.get("name", ""))
        parts.append(phase.get("summary", ""))
    return "\n".join(p for p in parts if p)


def _score_memory(probe: dict, facts: tuple[Fact, ...]) -> dict[str, Any]:
    """Run one MemoryDriftMetric per live fact and fold them into columns."""
    if not facts:
        return {
            "memory_drift_mean": "",
            "memory_drift_project_name_score": "",
            "persist_facts_total": 0,
            "persist_facts_recalled": 0,
            "stale_facts_total": 0,
            "stale_facts_present": 0,
            "memory_drift_detail": "",
        }

    results = run_all_metrics([MemoryDriftMetric(f) for f in facts], probe)
    by_key = {f.key: r for f, r in zip(facts, results)}

    persist = [f for f in facts if f.expectation == "persist"]
    stale = [f for f in facts if f.expectation == "superseded"]
    persist_scores = [by_key[f.key].score for f in persist]

    return {
        # The headline recall number is over facts that are *supposed* to
        # persist. Averaging stale facts in would reward forgetting.
        "memory_drift_mean": round(statistics.fmean(persist_scores), 4) if persist_scores else "",
        "memory_drift_project_name_score": by_key["project_name"].score
        if "project_name" in by_key
        else "",
        "persist_facts_total": len(persist),
        "persist_facts_recalled": int(sum(persist_scores)),
        "stale_facts_total": len(stale),
        "stale_facts_present": int(sum(by_key[f.key].score for f in stale)),
        "memory_drift_detail": ";".join(f"{k}={int(r.score)}" for k, r in by_key.items()),
    }


def _score_attachment_recall(probe: dict, expected: bool) -> dict[str, Any]:
    """Did each planted marker survive into the answer?

    ``expected`` is False for the 0 KB baseline and for a rejected
    attachment: there, a marker could not possibly be recalled, and scoring
    it 0.0 would drag the curve down for a reason that is not memory.
    """
    if not expected:
        return {"attachment_recall_mean": "", "attachment_recall_detail": ""}

    # The marker's own surface forms are the needles; its ``key`` is only a
    # label. Passing the key as the fact would put words like "deadline"
    # into the search, and an estimation summary says "deadline" all the
    # time — the metric would then score the model's vocabulary rather than
    # whether the attachment reached it.
    results = run_all_metrics(
        [
            MemoryDriftMetric(
                m.needles[0],
                where=["response"],
                aliases=m.needles[1:],
                name=f"attachment_{m.key}",
            )
            for m in MARKERS
        ],
        probe,
    )
    scores = [r.score for r in results]
    return {
        "attachment_recall_mean": round(statistics.fmean(scores), 4),
        "attachment_recall_detail": ";".join(
            f"{m.position}={int(r.score)}" for m, r in zip(MARKERS, results)
        ),
    }


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------
@dataclass
class RunConfig:
    scenarios: list[Scenario]
    attachment_sizes_kb: list[int]
    repeats: int
    max_turns: int
    latency_budget_ms: int
    cost_budget_usd: float
    pdf_paths: dict[int, Path]


def _run_one_turn(
    client: Client,
    session_id: str,
    transcript: str,
    pdf: Path | None,
    scenario: Scenario,
    turn_index: int,
    config: RunConfig,
    *,
    attachment_expected: bool,
) -> dict[str, Any]:
    """One estimate + one snapshot read + all metrics, as a flat row."""
    try:
        response = client.estimate(session_id, transcript, pdf)
    except Exception as exc:  # noqa: BLE001 — a dead turn must not kill the run
        return {
            "turn_index": turn_index,
            "session_id": session_id,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }

    snapshot = client.snapshot(session_id)
    observed = snapshot.get("last_turn") or {}
    result = response.get("result", {})

    # One probe object serves both memory drift (reads the session's slots)
    # and attachment recall (reads this turn's answer).
    probe = {**snapshot, "response_summary": _response_text(result)}

    budget_results = run_all_metrics(
        [
            LatencyBudgetMetric(config.latency_budget_ms),
            CostBudgetMetric(config.cost_budget_usd),
        ],
        observed,
    )
    row: dict[str, Any] = {k: observed.get(k, "") for k in COLUMNS if k in observed}
    row.update(
        {
            "turn_index": observed.get("turn_index", turn_index),
            "session_id": session_id,
            "confidence_pct": result.get("confidence_pct", ""),
            "total_cost_eur": result.get("total_cost_eur", ""),
            "error": "",
        }
    )
    for metric_result in budget_results:
        row.update(
            {
                f"{metric_result.name}_score": metric_result.score,
                f"{metric_result.name}_passed": int(metric_result.passed),
            }
        )
    row.update(_score_memory(probe, scenario.facts_live_at(turn_index)))
    row.update(_score_attachment_recall(probe, attachment_expected))
    return row


def _turn_sweep(client: Client, config: RunConfig) -> Iterator[dict[str, Any]]:
    """Conversation length is the variable; no attachments anywhere."""
    for scenario in config.scenarios:
        for repeat in range(1, config.repeats + 1):
            session_id = client.create_session()
            cumulative = 0.0
            for turn in scenario.turns[: config.max_turns]:
                row = _run_one_turn(
                    client,
                    session_id,
                    turn.transcript,
                    None,
                    scenario,
                    turn.index,
                    config,
                    attachment_expected=False,
                )
                cumulative += float(row.get("cost_usd") or 0.0)
                row.update(
                    {
                        "sweep": "turns",
                        "scenario": scenario.key,
                        "repeat": repeat,
                        "attachment_size_kb": 0,
                        "cum_cost_usd": round(cumulative, 8),
                    }
                )
                yield row


def _attachment_sweep(client: Client, config: RunConfig) -> Iterator[dict[str, Any]]:
    """Attachment size is the variable; every session is exactly one turn
    and carries the same opening transcript."""
    for scenario in config.scenarios:
        for size_kb in config.attachment_sizes_kb:
            pdf = config.pdf_paths.get(size_kb) if size_kb else None
            for repeat in range(1, config.repeats + 1):
                session_id = client.create_session()
                row = _run_one_turn(
                    client,
                    session_id,
                    scenario.opening_transcript,
                    pdf,
                    scenario,
                    1,
                    config,
                    # A rejected oversize attachment never reaches the model,
                    # so its markers are not scorable (see _score_attachment_recall).
                    attachment_expected=bool(size_kb),
                )
                if row.get("attachments_rejected"):
                    row.update({"attachment_recall_mean": "", "attachment_recall_detail": "rejected"})
                row.update(
                    {
                        "sweep": "attachments",
                        "scenario": scenario.key,
                        "repeat": repeat,
                        "attachment_size_kb": size_kb,
                        "cum_cost_usd": round(float(row.get("cost_usd") or 0.0), 8),
                    }
                )
                yield row


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAG stress runner (Session 6)")
    parser.add_argument("--http", default=None, help="base URL; omit to run in-process")
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100", help="KB of extracted text")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--latency-budget-ms", type=int, default=4000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.005)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default="evals/stress/results.csv")
    parser.add_argument(
        "--sweeps",
        default="turns,attachments",
        help="which sweeps to run; useful for resuming or for a cheap pilot",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sizes = [int(s) for s in args.attachment_sizes.split(",") if s.strip()]
    config = RunConfig(
        scenarios=get_scenarios([s.strip() for s in args.scenarios.split(",") if s.strip()]),
        attachment_sizes_kb=sizes,
        repeats=args.repeats,
        max_turns=args.max_turns,
        latency_budget_ms=args.latency_budget_ms,
        cost_budget_usd=args.cost_budget_usd,
        pdf_paths=ensure_corpus(tuple(s for s in sizes if s > 0)),
    )
    client = Client.over_http(args.http, args.timeout) if args.http else Client.in_process()
    wanted = [s.strip() for s in args.sweeps.split(",") if s.strip()]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    written = failed = 0
    spent = 0.0

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()

        sweeps = []
        if "turns" in wanted:
            sweeps.append(_turn_sweep(client, config))
        if "attachments" in wanted:
            sweeps.append(_attachment_sweep(client, config))

        for sweep in sweeps:
            for row in sweep:
                writer.writerow({k: row.get(k, "") for k in COLUMNS})
                # Flushed per row: a run interrupted at turn 140 of 225
                # should leave 140 usable rows, not an empty file.
                handle.flush()
                written += 1
                spent += float(row.get("cost_usd") or 0.0)
                if row.get("error"):
                    failed += 1
                    print(
                        f"  [{written:>3}] {row['sweep']:<11} {row['scenario']:<13} "
                        f"r{row['repeat']} t{row['turn_index']:<2} ERROR {row['error']}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  [{written:>3}] {row['sweep']:<11} {row['scenario']:<13} "
                        f"r{row['repeat']} t{row['turn_index']:<2} "
                        f"{row['attachment_size_kb']:>3}KB "
                        f"{row['latency_ms']:>6}ms  ${row['cost_usd']:<10} "
                        f"win={row['messages_in_window']:<2} anc={row['anchors_count']:<2} "
                        f"sum={row['summary_chars']:<5} recall={row['memory_drift_mean']}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - started
    print(
        f"\n{written} rows ({failed} failed) in {elapsed / 60:.1f} min, "
        f"${spent:.4f} spent -> {output}"
    )
    return 1 if failed and failed == written else 0


if __name__ == "__main__":
    raise SystemExit(main())
