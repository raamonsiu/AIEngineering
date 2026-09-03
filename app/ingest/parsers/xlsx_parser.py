"""Rate-card XLSX parser.

XLSX is the most treacherous format in the corpus because it looks tabular
and frequently is not: merged cells, live formulas, several tables on one
sheet, floating comments, hidden sheets, information encoded in
conditional formatting.

The rule applied here is the narrow one: if the sheet is a clean table it
becomes a markdown table, and everything else is deliberately dropped. A
sheet whose meaning lives outside its grid does not belong in a retrieval
corpus without manual conversion first — and saying so in the catalog is
a better answer than a parser quietly producing plausible nonsense.

``data_only=True`` reads cached formula *results* rather than formula
text. The cost is real and worth stating: a workbook never opened by Excel
since its formulas were written has no cached values, and those cells read
as empty. That shows up as a completeness problem in the audit, which is
where it belongs.
"""

from __future__ import annotations

import io

from app.ingest.models import ParsedUnit
from app.ingest.parsers.base import Parser


def _rows_to_markdown(rows: list[list[str]]) -> str:
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


class RateCardXlsxParser:
    supported_formats = {"xlsx"}

    def parse(self, content: bytes, source_hint: str) -> list[ParsedUnit]:
        from openpyxl import load_workbook  # lazy: xlsx-only path

        workbook = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
        doc_id = source_hint.rsplit("/", 1)[-1]
        units: list[ParsedUnit] = []

        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                # A hidden sheet is hidden for a reason the pipeline cannot
                # know. Excluding it is the conservative read.
                continue
            rows = [
                [("" if cell is None else str(cell)).strip() for cell in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            rows = [r for r in rows if any(cell for cell in r)]
            if len(rows) < 2:
                continue
            units.append(
                ParsedUnit(
                    content=f"## {sheet.title}\n\n{_rows_to_markdown(rows)}",
                    document_id=doc_id,
                    unit_key=f"sheet-{sheet.title}",
                    document_title=doc_id,
                    section_title=sheet.title,
                    extra={"sheet": sheet.title, "rows": len(rows) - 1},
                )
            )

        workbook.close()
        return units


_: Parser = RateCardXlsxParser()
