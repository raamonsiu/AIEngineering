"""Budget JSON parser.

JSON is the "easy" format only in the sense that nothing has to be
extracted — the hard decision is what textual shape the records take on
the way into the index, and that decision is not generic.

``json.dumps`` of the whole record produces noisy embeddings: it mixes
technical keys (``client_code``, ``hours_estimated``) with semantic values
in one undifferentiated string, and the key names dominate the vector for
every record equally, which is the opposite of discriminating between
them. Dumping only the values loses what each one meant. So the parser
renders **structured markdown**: important keys are promoted to headings,
values are written as prose, and the phase breakdown becomes a table.
That requires a parser that knows this schema — which is the point. A
schema-blind JSON parser cannot make this choice, so it makes the bad one.

This is a ``TabularParser``: its real intermediate representation is a
list of records destined for the pandas + Pandera path. ``parse`` renders
straight through and exists for callers that want raw passthrough.
"""

from __future__ import annotations

import json

from app.ingest.models import ParsedUnit
from app.ingest.parsers.base import TabularParser


def _fmt_money(value, currency: str | None) -> str:
    try:
        return f"{float(value):,.2f} {currency or ''}".strip()
    except (TypeError, ValueError):
        # A value that will not cast is left verbatim rather than blanked:
        # the cleaning layer decides what to do with it, and hiding it
        # here would mean the validator never sees the problem.
        return f"{value} {currency or ''}".strip()


class BudgetJsonParser:
    supported_formats = {"json"}

    def to_records(self, content: bytes, source_hint: str) -> list[dict]:
        """Raw records, untouched. Normalisation belongs to the cleaning
        layer, not here: a parser that quietly fixed a date format would
        hide a data-quality signal the audit is supposed to surface."""
        payload = json.loads(content.decode("utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            record.setdefault("_source_file", source_hint)
        return records

    def render_record(self, record: dict) -> ParsedUnit:
        budget_id = str(record.get("budget_id", "")) or "unknown"
        client = record.get("client_name") or "unknown client"
        currency = record.get("currency")

        lines = [
            f"# Presupuesto {budget_id}",
            "",
            f"Cliente: {client}.",
            f"Tipo de proyecto: {record.get('project_type', 'no especificado')}.",
            f"Importe total: {_fmt_money(record.get('total_amount'), currency)}.",
            f"Horas estimadas: {record.get('hours_estimated', 'no especificadas')}.",
            f"Estado: {record.get('status', 'desconocido')}.",
            f"Fecha de firma: {record.get('signed_at', 'sin fecha')}.",
        ]

        phases = record.get("phases") or []
        if phases:
            lines += [
                "",
                "## Desglose por fases",
                "",
                "| Fase | Horas | Coste (EUR) |",
                "|------|-------|-------------|",
            ]
            lines += [
                f"| {p.get('name', '')} | {p.get('hours', '')} | {p.get('cost_eur', '')} |"
                for p in phases
            ]

        return ParsedUnit(
            content="\n".join(lines),
            document_id=budget_id,
            document_title=f"Presupuesto {budget_id} — {client}",
            document_author=record.get("account_manager"),
            extra={
                "status": record.get("status"),
                "client_code": record.get("client_code"),
                "currency": currency,
                "source_file": record.get("_source_file"),
            },
        )

    def parse(self, content: bytes, source_hint: str) -> list[ParsedUnit]:
        return [self.render_record(r) for r in self.to_records(content, source_hint)]


_: TabularParser = BudgetJsonParser()  # structural conformance check
