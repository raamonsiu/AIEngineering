"""Tests for the cleaning and validation layer.

One test per family of dirt, plus the routing policy. The families are
the vocabulary the whole layer is organised around, so if one of them
has no test, the layer has a hole with a name.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ingest.cleaning.budgets import (
    canonical_client_key,
    clean_budget_records,
    parse_amount,
    parse_date,
    unmask_nulls,
)
from app.ingest.cleaning.policy import validate_with_policy
from app.ingest.cleaning.schemas import BudgetRecord
from app.ingest.cleaning.text import clean_text_units, normalise_text
from app.ingest.models import ParsedUnit


def _record(**overrides) -> dict:
    base = {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme Corp",
        "total_amount": 80000,
        "currency": "EUR",
        "hours_estimated": 1280,
        "signed_at": "2024-03-15",
        "status": "signed",
    }
    return {**base, **overrides}


# ----------------------------------------------------------------------
# Family 1: heterogeneidad de formato
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "written", [80000, 80000.0, "80000", "80.000,00", "80,000.00", "80000 €"]
)
def test_the_same_amount_written_six_ways_converges(written) -> None:
    assert parse_amount(written) == 80000.0


@pytest.mark.parametrize("written", ["2024-03-15", "15/03/2024", "Mar 15 2024", "15-03-2024"])
def test_the_same_date_written_four_ways_converges(written) -> None:
    assert str(parse_date(written).date()) == "2024-03-15"


def test_currency_variants_collapse_to_the_canonical_code() -> None:
    df = clean_budget_records(
        pd.DataFrame([_record(budget_id=f"BUDGET-2024-000{i}", currency=c)
                      for i, c in enumerate(["EUR", "eur", "€", "euros"], start=1)])
    )

    assert set(df["currency"]) == {"EUR"}


def test_client_variants_share_a_key_without_losing_the_original_name() -> None:
    """The scalpel, not the chainsaw: group by a key, keep the display value."""
    df = clean_budget_records(
        pd.DataFrame([
            _record(budget_id="BUDGET-2024-0001", client_name="ACME Corp."),
            _record(budget_id="BUDGET-2024-0002", client_name="Acme Corp"),
            _record(budget_id="BUDGET-2024-0003", client_name="acme corp"),
        ])
    )

    assert df["client_key"].nunique() == 1
    # Casing and punctuation survive, because they mean things elsewhere.
    assert set(df["client_name"]) == {"ACME Corp.", "Acme Corp", "acme corp"}


def test_canonical_key_ignores_legal_suffixes() -> None:
    assert canonical_client_key("Acme S.L.") == canonical_client_key("ACME")


# ----------------------------------------------------------------------
# Family 2: duplicados divergentes
# ----------------------------------------------------------------------
def test_a_divergent_duplicate_is_resolved_and_recorded() -> None:
    df = clean_budget_records(
        pd.DataFrame([
            _record(total_amount=80000, signed_at="2024-03-15"),
            _record(total_amount=82500, signed_at="2024-06-01"),
        ])
    )

    assert len(df) == 1
    assert df.iloc[0]["total_amount"] == 82500.0  # latest signature wins
    # Recorded, not silently resolved: a rising count is how you learn an
    # upstream export started running twice.
    assert df.attrs["divergent_budget_ids"] == ["BUDGET-2024-0001"]


def test_an_identical_duplicate_is_not_reported_as_divergent() -> None:
    df = clean_budget_records(pd.DataFrame([_record(), _record()]))

    assert len(df) == 1
    assert df.attrs["divergent_budget_ids"] == []


# ----------------------------------------------------------------------
# Family 3: nulos disfrazados
# ----------------------------------------------------------------------
@pytest.mark.parametrize("disguise", ["N/A", "n/a", "-", "", "  ", "TBD", "pendiente", "unknown"])
def test_disguised_nulls_become_real_nulls(disguise) -> None:
    assert unmask_nulls(pd.Series([disguise])).isna().all()


def test_a_real_value_survives_unmasking() -> None:
    assert unmask_nulls(pd.Series(["Ana Ruiz"])).iloc[0] == "Ana Ruiz"


# ----------------------------------------------------------------------
# Family 4: fuera de rango
# ----------------------------------------------------------------------
def test_out_of_range_records_are_discarded_not_quarantined() -> None:
    df = clean_budget_records(
        pd.DataFrame([
            _record(budget_id="BUDGET-2024-0001"),
            _record(budget_id="BUDGET-2024-0002", total_amount=-50000),
            _record(budget_id="BUDGET-2024-0003", total_amount=99_000_000),
            _record(budget_id="BUDGET-2024-0004", signed_at="2099-01-01"),
        ])
    )

    result = validate_with_policy(df, BudgetRecord)

    assert list(result.valid["budget_id"]) == ["BUDGET-2024-0001"]
    assert result.report["discarded"] == 3
    assert result.report["quarantined"] == 0


def test_a_recoverable_gap_is_quarantined_with_its_reason() -> None:
    """Missing-but-otherwise-complete is for a human, not the bin."""
    df = clean_budget_records(pd.DataFrame([_record(client_name="N/A")]))

    result = validate_with_policy(df, BudgetRecord)

    assert result.report["quarantined"] == 1
    assert result.report["discarded"] == 0
    assert "client_name" in result.quarantined.iloc[0]["validation_reasons"]


def test_a_non_canonical_identifier_is_contamination() -> None:
    df = clean_budget_records(pd.DataFrame([_record(budget_id="BDG_2024_12")]))

    result = validate_with_policy(df, BudgetRecord)

    assert result.report["discarded"] == 1


def test_cross_column_rule_catches_individually_plausible_fields() -> None:
    """status='signed' and total=0 are each fine; together they are not."""
    df = clean_budget_records(pd.DataFrame([_record(total_amount=0, status="signed")]))

    result = validate_with_policy(df, BudgetRecord)

    assert result.report["valid"] == 0


def test_permissive_mode_changes_the_consequence_not_the_contract() -> None:
    df = clean_budget_records(pd.DataFrame([_record(total_amount=-50000)]))

    strict = validate_with_policy(df, BudgetRecord, mode="strict")
    permissive = validate_with_policy(df, BudgetRecord, mode="permissive")

    assert strict.report["discarded"] == 1 and strict.report["quarantined"] == 0
    assert permissive.report["discarded"] == 0 and permissive.report["quarantined"] == 1
    # Neither mode calls it valid: the contract is identical.
    assert strict.report["valid"] == permissive.report["valid"] == 0


def test_lazy_validation_reports_every_failure_not_just_the_first() -> None:
    """Without lazy=True the policy could not route by failure type."""
    df = clean_budget_records(
        pd.DataFrame([
            _record(budget_id="BDG_BAD"),
            _record(budget_id="BUDGET-2024-0002", total_amount=-1),
            _record(budget_id="BUDGET-2024-0003", client_name="N/A"),
        ])
    )

    result = validate_with_policy(df, BudgetRecord)

    assert len(result.report["failure_breakdown"]) >= 2
    assert result.report["total"] == 3


def test_a_clean_batch_produces_no_schema_level_failures() -> None:
    """Guards the dtype contract: with coerce=False, the cleaning layer is
    responsible for emitting exactly the declared types."""
    df = clean_budget_records(pd.DataFrame([_record()]))

    result = validate_with_policy(df, BudgetRecord)

    assert result.report["valid"] == 1
    assert result.report.get("schema_level_failures", 0) == 0


def test_an_empty_batch_is_not_an_error() -> None:
    result = validate_with_policy(pd.DataFrame(), BudgetRecord)

    assert result.report == {"total": 0, "valid": 0}


# ----------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------
def test_text_normalisation_removes_only_accidental_heterogeneity() -> None:
    cleaned = normalise_text("Hola   mundo\x00\n\n\n\nSegundo  párrafo   ")

    assert cleaned == "Hola mundo\n\nSegundo párrafo"
    # Case and accents are untouched: they carry meaning.
    assert "párrafo" in cleaned


def test_text_cleaning_drops_artefacts_and_keeps_content() -> None:
    units = [
        ParsedUnit(content="Página 3 de 12", document_id="a"),
        ParsedUnit(content="   ", document_id="b"),
        ParsedUnit(content="corto", document_id="c"),
        ParsedUnit(content="Este párrafo es suficientemente largo para sobrevivir.", document_id="d"),
    ]

    kept, report = clean_text_units(units)

    assert [u.document_id for u in kept] == ["d"]
    assert (report.dropped_artefact, report.dropped_empty, report.dropped_too_short) == (1, 1, 1)


def test_a_placeholder_turn_is_flagged_not_dropped_when_long_enough() -> None:
    """A transcript turn that literally says 'pendiente' is a real thing
    to have said; the tabular rule does not apply to prose."""
    units = [ParsedUnit(content="pendiente", document_id="t1")]

    _kept, report = clean_text_units(units)

    assert report.placeholder_units == ["t1"]
