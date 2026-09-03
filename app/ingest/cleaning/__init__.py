"""The one place data invariants are enforced.

Split in two halves on purpose. ``budget_records``/``text`` *normalise*:
they transform what can be transformed and decide nothing, leaving nulls
for someone else to rule on. ``schemas`` + ``policy`` *validate*: they
state the contract and route each failure to valid, quarantine or
discard.

Separating them is what lets the policy change (quarantine in
development, discard in production) without touching a single regex — and
what keeps the contract itself identical in both.
"""

from app.ingest.cleaning.budgets import clean_budget_records
from app.ingest.cleaning.policy import ValidationResult, validate_with_policy
from app.ingest.cleaning.schemas import BudgetRecord
from app.ingest.cleaning.text import clean_text_units, normalise_text

__all__ = [
    "BudgetRecord",
    "ValidationResult",
    "clean_budget_records",
    "clean_text_units",
    "normalise_text",
    "validate_with_policy",
]
