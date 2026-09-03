"""Signed-contract PDF parser.

PDF is a presentation format, not a content one: structure is implicit in
positions and font sizes. The project's corpus is born-digital contracts
with a real text layer, so the native cascade in
``app.ingest.parsers.extraction`` (pypdf, falling back to PyMuPDF) is the
right tool and ``unstructured`` with ``hi_res`` is not: that path costs an
order of magnitude more latency and compute to recover tables and run OCR
that this corpus does not need. The catalog records the PDF class per
source, so the decision is taken once, not per document.

Units are pages. A page is the smallest thing a contract citation can
point at ("cláusula 4, página 2"), and the extractor already marks page
boundaries, so no structure is invented that the format did not supply.
"""

from __future__ import annotations

import re

from app.ingest.models import ParsedUnit
from app.ingest.parsers.base import Parser
from app.ingest.parsers.extraction import extract_attachment

_PAGE_MARKER = re.compile(r"^--- Page (\d+) ---$", re.MULTILINE)


class ContractPdfParser:
    supported_formats = {"pdf"}

    def parse(self, content: bytes, source_hint: str) -> list[ParsedUnit]:
        doc_id = source_hint.rsplit("/", 1)[-1]
        result = extract_attachment(doc_id, content)
        if not result.ok:
            # A failed extraction returns nothing rather than an empty
            # unit. An empty Document would travel the whole pipeline and
            # be embedded as a vector meaning nothing, which is worse than
            # an absence the orchestrator can count and report.
            return []

        pieces = _PAGE_MARKER.split(result.text)
        units: list[ParsedUnit] = []
        # split() yields [preamble, page_no, body, page_no, body, ...]
        for page_no, body in zip(pieces[1::2], pieces[2::2]):
            text = body.strip()
            if not text:
                continue
            units.append(
                ParsedUnit(
                    content=text,
                    document_id=doc_id,
                    unit_key=f"page-{int(page_no):04d}",
                    document_title=doc_id,
                    page_number=int(page_no),
                    extra={"extraction_method": result.method},
                )
            )
        if not units and result.text.strip():
            units.append(
                ParsedUnit(
                    content=result.text.strip(),
                    document_id=doc_id,
                    unit_key="whole",
                    document_title=doc_id,
                    extra={"extraction_method": result.method, "pagination": "unavailable"},
                )
            )
        return units


_: Parser = ContractPdfParser()
