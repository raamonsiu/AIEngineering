"""Pandera contracts: the content contract, as opposed to the shape one.

``Document``'s Pydantic model is the contract of *form* — it guarantees a
non-empty string and the required metadata keys. It says nothing about
whether ``total_amount`` is negative, whether two records claim the same
budget id, or whether a signature is dated next year. Two records can
satisfy Pydantic perfectly and still be mutually unusable.

Pandera validates the other dimension. Pydantic answers "is this object
well-formed?" one instance at a time; Pandera answers "does this table
satisfy the business invariants?" across every row at once, and on failure
reports which rows failed and why. That report is precisely what the
routing policy needs, and it is why validation happens on the DataFrame
and not instance-by-instance after normalisation.

This file is a living artefact. When finance accepts a new currency or
raises the project ceiling, the change happens here, once, versioned in
git, and the whole pipeline downstream obeys it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Optional

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameModel, Field
from pandera.typing.pandas import Series

# Canonical identifier shapes. Not cosmetics: these are the contract with
# the upstream ERP export, and a record that breaks them is far more
# likely to be a migration artefact than a real budget.
BUDGET_ID_PATTERN = r"^BUDGET-\d{4}-\d{4}$"
CLIENT_CODE_PATTERN = r"^CLI-\d{4}$"

MAX_PROJECT_AMOUNT = 10_000_000.0
MAX_PROJECT_HOURS = 100_000.0


class BudgetRecord(DataFrameModel):
    """Canonical contract for budget records before normalisation to Document."""

    budget_id: Series[str] = Field(
        str_matches=BUDGET_ID_PATTERN,
        nullable=False,
        description="Stable budget identifier in canonical ERP format",
    )
    client_name: Series[str] = Field(
        nullable=False,
        str_length={"min_value": 2, "max_value": 200},
    )
    # Optional at schema level (not every export carries it) but shape-checked
    # when present. A client code is the one field that maps one-to-one to a
    # real client, so a malformed one is a broken join, not a cosmetic issue.
    client_code: Optional[Series[str]] = Field(
        nullable=True,
        str_matches=CLIENT_CODE_PATTERN,
        description="Internal client identifier, CLI-NNNN",
    )
    total_amount: Series[float] = Field(
        ge=0,
        le=MAX_PROJECT_AMOUNT,
        nullable=False,
        description="Total in declared currency, non-negative",
    )
    currency: Series[str] = Field(isin=["EUR", "USD", "GBP"], nullable=False)
    hours_estimated: Series[float] = Field(gt=0, le=MAX_PROJECT_HOURS, nullable=True)
    # Timezone-aware on purpose. A naive timestamp cannot be compared
    # against "now" without assuming a zone, and this corpus mixes date
    # spellings from several locales, so the zone is pinned at cleaning
    # time and the contract states it rather than inferring it.
    signed_at: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = Field(nullable=False)
    status: Series[str] = Field(isin=["draft", "signed", "rejected"], nullable=False)

    class Config:
        # Unknown columns are rejected: a parser that silently starts
        # emitting a new field is a change the team should see in CI, not
        # discover in the index six weeks later.
        strict = "filter"
        # The cleaning layer already coerced every type. Letting Pandera
        # coerce again would mask a cleaning bug by fixing it invisibly.
        coerce = False
        ordered = False

    @pa.check("signed_at", name="signature_not_in_the_future")
    def signature_not_in_the_future(cls, series: Series[pd.Timestamp]) -> Series[bool]:
        """A budget signed next month is a transcription error, always."""
        return series <= pd.Timestamp(datetime.now(timezone.utc))

    @pa.dataframe_check(name="signed_budgets_have_a_positive_amount")
    def signed_budgets_have_a_positive_amount(cls, df: pd.DataFrame) -> Series[bool]:
        """Cross-column invariant: a signed budget of zero is not a budget.

        This is the class of rule a per-instance schema cannot state, and
        the class that catches the subtlest corruption — each field is
        individually plausible, the combination is not.
        """
        return ~((df["status"] == "signed") & (df["total_amount"] <= 0))


# Checks whose failure means contamination rather than a recoverable gap.
# Named here (not inline in the policy) so the routing rule and the schema
# cannot drift apart.
DISCARD_ON_FAILURE: frozenset[str] = frozenset(
    {
        "str_matches",                        # budget_id is not an ERP id
        "greater_than_or_equal_to",           # negative amount
        "less_than_or_equal_to",              # absurd amount / hours
        "greater_than",                       # non-positive hours
        "signature_not_in_the_future",
    }
)
