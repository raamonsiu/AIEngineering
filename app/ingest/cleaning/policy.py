"""What happens to a record that fails validation.

Three answers, and which one applies depends on the kind of failure, not
on the severity of the mood the team is in that day:

- **repair** — recoverable without semantic loss. Already done upstream by
  ``clean_budget_records``; by the time a record reaches here, the
  automatic repairs have been attempted and this layer does not retry.
- **quarantine** — serious, but the record might be useful after a human
  looks. A missing client name on an otherwise complete budget. These do
  not reach the index, and they are not thrown away either: they wait,
  with the reason attached, for arbitration.
- **discard** — contamination with no rescue value. A budget id that is
  not a budget id, a negative total, a signature dated next year. Logged
  in detail, then dropped. Keeping them would cost quarantine space and
  buy nothing.

The policy lives apart from the schema on purpose: in development you may
want everything quarantined so you can look at it, and in production you
want contamination discarded. What must NOT differ between the two is the
contract — relaxing *that* in development is how a team discovers on
deploy day that half its corpus never satisfied the rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import pandera.pandas as pa

from app.ingest.cleaning.schemas import DISCARD_ON_FAILURE

logger = logging.getLogger(__name__)

FailureMode = Literal["strict", "permissive"]


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    quarantined: pd.DataFrame
    discarded: pd.DataFrame
    report: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        total = self.report.get("total", 0)
        return self.report.get("valid", 0) / total if total else 0.0


def validate_with_policy(
    df: pd.DataFrame,
    schema: type[pa.DataFrameModel],
    *,
    mode: FailureMode = "strict",
) -> ValidationResult:
    """Validate a dataframe and route its failures.

    ``lazy=True`` is what makes the whole thing possible: without it
    Pandera raises on the first error and the policy would be blind to
    every other one, so routing by failure type could not exist. With it,
    the full failure set comes back at once and each row can be sent
    where its own failure says it belongs.

    ``mode="permissive"`` routes everything to quarantine instead of
    discarding. The contract is identical in both modes; only the
    consequence differs.
    """
    if df.empty:
        return ValidationResult(df, df.copy(), df.copy(), {"total": 0, "valid": 0})

    try:
        valid = schema.validate(df, lazy=True)
        report = {
            "total": len(df), "valid": len(valid), "quarantined": 0, "discarded": 0,
            "failure_breakdown": {},
        }
        return ValidationResult(valid, df.iloc[0:0].copy(), df.iloc[0:0].copy(), report)
    except pa.errors.SchemaErrors as exc:
        return _route_failures(df, exc, mode=mode)


def _route_failures(
    df: pd.DataFrame, exc: pa.errors.SchemaErrors, *, mode: FailureMode
) -> ValidationResult:
    cases = exc.failure_cases
    # Schema-level failures (an entirely missing column, say) carry no row
    # index. They are not routable per record and must not silently drop
    # the batch, so they are surfaced in the report instead.
    indexed = cases.dropna(subset=["index"]) if "index" in cases else cases.iloc[0:0]
    schema_level = len(cases) - len(indexed)

    failed_index = pd.Index(indexed["index"].astype(int).unique()) if len(indexed) else pd.Index([])
    discard_index = (
        pd.Index(
            indexed.loc[indexed["check"].astype(str).str.split("(").str[0]
                        .isin(DISCARD_ON_FAILURE), "index"].astype(int).unique()
        )
        if len(indexed) and mode == "strict"
        else pd.Index([])
    )
    quarantine_index = failed_index.difference(discard_index)
    valid_index = df.index.difference(failed_index)

    result = ValidationResult(
        valid=df.loc[valid_index].copy(),
        quarantined=_annotate(df.loc[quarantine_index], indexed, "quarantined"),
        discarded=_annotate(df.loc[discard_index], indexed, "discarded"),
        report={
            "total": len(df),
            "valid": len(valid_index),
            "quarantined": len(quarantine_index),
            "discarded": len(discard_index),
            "schema_level_failures": schema_level,
            "failure_breakdown": cases["check"].value_counts().to_dict(),
            "mode": mode,
        },
    )
    logger.warning(
        "ingest validation: total=%d valid=%d quarantined=%d discarded=%d",
        result.report["total"], result.report["valid"],
        result.report["quarantined"], result.report["discarded"],
    )
    return result


def _annotate(subset: pd.DataFrame, cases: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Attach the reason to each routed record.

    A quarantine table without reasons is a pile of rows nobody can
    arbitrate: the reviewer would have to re-run validation to find out
    why each one is there.
    """
    out = subset.copy()
    if out.empty:
        out["validation_outcome"] = pd.Series(dtype="string")
        out["validation_reasons"] = pd.Series(dtype="string")
        return out

    reasons = (
        cases.assign(index=cases["index"].astype(int))
        .groupby("index")
        .apply(lambda g: "; ".join(sorted({f"{c}:{k}" for c, k in zip(g["column"], g["check"])})),
               include_groups=False)
    )
    out["validation_outcome"] = outcome
    out["validation_reasons"] = out.index.map(reasons).astype("string")
    return out
