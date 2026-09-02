"""Cleaning for the non-tabular half of the corpus.

Transcripts, proposals and contracts have no columns, so Pandera has
nothing to grip. The same four families of dirt still apply, but the
techniques are different: encoding checks, placeholder detection, minimum
length, whitespace that carries no meaning.

The bar is deliberately lower than the tabular one. A budget record with
a missing client is unusable; a transcript turn that is slightly odd is
still evidence of what was said in a meeting. So this layer drops only
what is positively worthless — empty units, pure boilerplate, extraction
artefacts — and annotates the rest instead of judging it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.ingest.cleaning.budgets import NULL_PLACEHOLDERS
from app.ingest.models import ParsedUnit

# Below this, a "unit" is a page number, a header fragment or an
# extraction artefact rather than content worth embedding.
MIN_CONTENT_CHARS = 25

_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Control characters that survive PDF and DOCX extraction and embed as noise.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_ARTEFACT = re.compile(r"^\s*(p[áa]gina\s+\d+\s*(de\s+\d+)?|\d+\s*/\s*\d+)\s*$", re.IGNORECASE)


@dataclass
class TextCleaningReport:
    kept: int = 0
    dropped_empty: int = 0
    dropped_too_short: int = 0
    dropped_artefact: int = 0
    placeholder_units: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return self.dropped_empty + self.dropped_too_short + self.dropped_artefact


def normalise_text(text: str) -> str:
    """Scalpel, not chainsaw.

    Only accidental heterogeneity is removed: control characters, runs of
    spaces, runs of blank lines, and inconsistent Unicode composition
    (``NFKC`` so "café" written two ways becomes one token rather than
    two). Case is left alone, accents are left alone, and intentional
    paragraph structure survives, because normalising those would erase
    meaning to buy tidiness.
    """
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL.sub("", cleaned)
    cleaned = _WS.sub(" ", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    return _BLANK_LINES.sub("\n\n", cleaned).strip()


def is_placeholder(text: str) -> bool:
    """A unit whose entire content is a disguised null."""
    return text.strip().lower() in NULL_PLACEHOLDERS


def clean_text_units(units: list[ParsedUnit]) -> tuple[list[ParsedUnit], TextCleaningReport]:
    report = TextCleaningReport()
    kept: list[ParsedUnit] = []

    for unit in units:
        content = normalise_text(unit.content)
        if not content:
            report.dropped_empty += 1
            continue
        if _PAGE_ARTEFACT.match(content):
            report.dropped_artefact += 1
            continue
        if is_placeholder(content):
            # Flagged, not dropped: a transcript turn that is literally
            # "pendiente" is a real (and informative) thing to have said.
            report.placeholder_units.append(unit.document_id)
        if len(content) < MIN_CONTENT_CHARS:
            report.dropped_too_short += 1
            continue

        kept.append(unit.model_copy(update={"content": content}))
        report.kept += 1

    return kept, report
