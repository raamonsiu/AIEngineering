"""Normalisation of budget records, before any validation decides anything.

This is one half of the cleaning layer and it deliberately decides
nothing: it transforms what can be transformed and leaves the rest as
``NaN``/``NaT`` for the validation step to rule on. Keeping "make it
canonical" apart from "is it acceptable" is what lets the policy be
changed (quarantine vs discard) without touching a single regex.

Every coercion is permissive (``errors="coerce"``). A value that will not
parse becomes null rather than raising, because one malformed record in a
batch of eighty must not take the batch down — and a null is a thing the
validator can route, whereas an exception is a thing that loses the other
seventy-nine.

The steps run in a fixed order, and the order carries meaning: nulls are
unmasked before deduplication (so a disguised null cannot win a
"keep the most complete" comparison), and dates are parsed before dedup
(so "keep the latest" has something to sort by).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import pandas as pd

# Strings that look like content and are not. Lowercased for comparison.
NULL_PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "-", "--", "n/a", "na", "n.a.", "null", "none", "unknown", "tbd",
     "pendiente", "pending", "por determinar", "?", "sin datos"}
)

_CURRENCY_MAP = {
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR", "eur.": "EUR",
    "usd": "USD", "$": "USD", "dolares": "USD", "dólares": "USD",
    "gbp": "GBP", "£": "GBP",
}

# Accepted date spellings, tried in order. Explicit rather than relying on
# pandas' inference: with 15/03/2024 the day-first vs month-first guess is
# genuinely ambiguous, and an inferred answer that is right 90% of the time
# silently corrupts the other 10%.
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%b %d %Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y")

# Legal forms, matched after punctuation has been flattened to spaces, so
# "S.L.", "S.L" and "SL" all arrive here as "s l" or "sl".
_LEGAL_FORM = re.compile(
    r"\b(s\s?[al]|inc|ltd|llc|plc|corp|corporation|company|gmbh|srl|bv|ag)\b"
)

_THOUSANDS_EU = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")  # 80.000,00
_THOUSANDS_EN = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")  # 80,000.00


def unmask_nulls(series: pd.Series) -> pd.Series:
    """Turn disguised nulls into real ones.

    These are the most dangerous family of the four: they pass every type
    check, embed as if they were content, and get presented to the user
    with the same authority as a real value. ``"pendiente"`` retrieved as
    an account manager's name is worse than no answer.
    """
    cleaned = series.astype("string").str.strip()
    return cleaned.where(~cleaned.str.lower().isin(NULL_PLACEHOLDERS), other=pd.NA)


def normalise_currency(series: pd.Series) -> pd.Series:
    lowered = series.astype("string").str.strip().str.lower()
    mapped = lowered.map(_CURRENCY_MAP)
    # Unmapped values are passed through uppercased rather than nulled:
    # an unknown currency code is a validation question, not a cleaning
    # one, and the schema's `isin` check is where it should be answered.
    return mapped.fillna(series.astype("string").str.strip().str.upper())


def parse_amount(value) -> float | None:
    """Parse an amount written as a number or as any of several strings.

    Both thousands conventions appear in this corpus, and they are
    ambiguous against each other: "80.000" is eighty thousand in Spanish
    notation and eighty in English. The regexes above resolve only the
    unambiguous shapes (a grouped pattern, with the right digit counts)
    and anything else falls through to a plain cast, so a guess is never
    made on a string that does not clearly declare its convention.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(" ", "").replace("€", "").replace("$", "")
    if not text or text.lower() in NULL_PLACEHOLDERS:
        return None
    if _THOUSANDS_EU.match(text):
        text = text.replace(".", "").replace(",", ".")
    elif _THOUSANDS_EN.match(text):
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, (datetime, pd.Timestamp)):
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    text = str(value).strip()
    if not text or text.lower() in NULL_PLACEHOLDERS:
        return pd.NaT
    for fmt in _DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt), tz="UTC")
        except ValueError:
            continue
    return pd.NaT


def canonical_client_key(value) -> str | None:
    """A grouping key for client names, NOT a replacement for them.

    "ACME Corp." / "Acme Corp" / "acme corp" are one client written three
    ways, and left apart they land in three regions of the vector space.
    But folding the display name itself would be the chainsaw: casing and
    punctuation carry meaning elsewhere. So the variants collapse into a
    key used for grouping and counting, while ``client_name`` keeps
    whatever the source actually said.
    """
    if value is None or pd.isna(value):
        return None
    # Order matters, and getting it wrong is subtle: punctuation is
    # flattened FIRST, so by the time legal forms are stripped "S.L." has
    # already become "s l". The suffix pattern therefore has to tolerate
    # the separating space rather than the dot it no longer contains.
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEGAL_FORM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip() or None


def clean_budget_records(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical cleaning sequence. Side-effect free."""
    out = df.copy()

    for column in ("client_name", "account_manager", "contact_email",
                   "contact_phone", "project_type", "status", "client_code"):
        if column in out:
            out[column] = unmask_nulls(out[column])

    if "currency" in out:
        out["currency"] = normalise_currency(out["currency"])
    if "total_amount" in out:
        out["total_amount"] = out["total_amount"].map(parse_amount).astype("Float64")
    if "hours_estimated" in out:
        # Cast to float even when every value is a whole number: the
        # schema declares float64 and `coerce=False`, so an int64 column
        # of perfectly good hours would be rejected on dtype alone.
        out["hours_estimated"] = pd.to_numeric(
            out["hours_estimated"], errors="coerce"
        ).astype("float64")
    if "signed_at" in out:
        # Resolution is pinned, not inherited. pandas picks the unit from
        # whatever it parsed (and its default changed between major
        # versions), so leaving it implicit makes the dtype a function of
        # the installed pandas rather than of this contract.
        out["signed_at"] = (
            out["signed_at"].map(parse_date).astype("datetime64[ns, UTC]")
        )
    if "status" in out:
        out["status"] = out["status"].str.lower()
    if "client_name" in out:
        out["client_key"] = out["client_name"].map(canonical_client_key)

    return _deduplicate(out)


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve divergent duplicates: same budget_id, different values.

    The rule is "keep the most recently signed version", and it is a
    business decision rather than a technical one — it encodes the
    assumption that a later record supersedes an earlier one, which is
    true for budget revisions and false for, say, append-only event logs.
    It belongs in the catalog's notes as much as in this function.

    Records whose divergence is resolved are *counted*, not silently
    dropped: a rising duplicate count is how you learn that an upstream
    export has started running twice.
    """
    if "budget_id" not in df or df.empty:
        return df

    fingerprint = df.apply(
        lambda row: hashlib.sha256(
            f"{row.get('budget_id')}|{row.get('total_amount')}|"
            f"{row.get('currency')}|{row.get('status')}".encode()
        ).hexdigest(),
        axis=1,
    )
    out = df.assign(_content_hash=fingerprint)

    divergent = (
        out.groupby("budget_id")["_content_hash"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    out.attrs["divergent_budget_ids"] = divergent
    out.attrs["exact_duplicates_dropped"] = int(
        len(out) - out.drop_duplicates(subset=["budget_id", "_content_hash"]).shape[0]
    )

    out = out.sort_values("signed_at", na_position="first")
    out = out.drop_duplicates(subset=["budget_id"], keep="last")
    kept_attrs = {k: v for k, v in df.attrs.items()}
    out = out.drop(columns=["_content_hash"])
    out.attrs.update(kept_attrs)
    out.attrs["divergent_budget_ids"] = divergent
    return out.reset_index(drop=True)
