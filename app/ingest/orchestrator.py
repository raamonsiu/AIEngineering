"""The offline indexing pipeline, end to end.

    catalog -> loader -> parser -> cleaning -> validation -> anonymisation
            -> normalizer -> Document[]

Two routes through the middle, chosen by the source's declared
``pipeline``:

- **tabular** — records become a DataFrame, get cleaned, and are validated
  against a Pandera contract that routes each failure to valid /
  quarantine / discard. Only the valid rows are rendered to text.
- **text** — units are normalised and filtered by the lighter text rules.
  There is no schema to validate against, and pretending otherwise would
  mean inventing invariants that prose does not have.

The catalog is obeyed, not consulted: a source marked ``exclude`` or
``review`` is not processed, however present its files are. That is the
entire point of writing the decision down — the ``rate_card`` is the
canonical example, officially the source of truth for rates and so stale
that including it would inject systematic error.

Nothing here is on a user's critical path. Minutes are fine, which is why
heavyweight work (spaCy NER over every document) belongs in this pipeline
and not in the query one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.ingest.anonymization.pseudonymizer import ConsistentPseudonymizer
from app.ingest.catalog import CatalogSource, DataCatalog, IngestionDecision
from app.ingest.cleaning.budgets import clean_budget_records
from app.ingest.cleaning.policy import ValidationResult, validate_with_policy
from app.ingest.cleaning.schemas import BudgetRecord
from app.ingest.cleaning.text import clean_text_units
from app.ingest.loaders.filesystem import FilesystemLoader
from app.ingest.models import Document, ParsedUnit
from app.ingest.normalizers.canonical import to_document
from app.ingest.parsers import DEFAULT_PARSERS, Parser

logger = logging.getLogger(__name__)


@dataclass
class SourceReport:
    """What the pipeline did to one source, stage by stage.

    Kept per source rather than aggregated because the useful question is
    never "how many documents did we index" but "which source started
    dropping records this week".
    """

    source_name: str
    decision: str
    files_seen: int = 0
    units_parsed: int = 0
    units_after_cleaning: int = 0
    documents_emitted: int = 0
    records_valid: int = 0
    records_quarantined: int = 0
    records_discarded: int = 0
    entities_anonymized: int = 0
    divergent_duplicates: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None


@dataclass
class IngestionRun:
    started_at: datetime
    finished_at: Optional[datetime] = None
    documents: list[Document] = field(default_factory=list)
    reports: list[SourceReport] = field(default_factory=list)
    quarantine: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def report_for(self, source_name: str) -> SourceReport:
        return next(r for r in self.reports if r.source_name == source_name)


class IngestionPipeline:
    def __init__(
        self,
        catalog: DataCatalog,
        *,
        corpus_root: Path,
        parsers: dict[str, Parser] | None = None,
        pseudonymizer: ConsistentPseudonymizer | None = None,
    ) -> None:
        self.catalog = catalog
        self.corpus_root = Path(corpus_root)
        self.parsers = parsers or DEFAULT_PARSERS
        self.loader = FilesystemLoader(self.corpus_root)
        self.pseudonymizer = pseudonymizer

    def run(self) -> IngestionRun:
        run = IngestionRun(started_at=datetime.now(timezone.utc))

        for source in self.catalog.sources:
            if source.decision is not IngestionDecision.INCLUDE:
                run.reports.append(
                    SourceReport(
                        source_name=source.name,
                        decision=source.decision.value,
                        skipped_reason=(
                            source.notes or f"catalog decision is '{source.decision.value}'"
                        ),
                    )
                )
                logger.info("ingest: skipping %s (%s)", source.name, source.decision.value)
                continue
            run = self._ingest_source(source, run)

        run.finished_at = datetime.now(timezone.utc)
        return run

    # ------------------------------------------------------------------
    def _ingest_source(self, source: CatalogSource, run: IngestionRun) -> IngestionRun:
        report = SourceReport(source_name=source.name, decision=source.decision.value)
        parser = self.parsers.get(source.format)
        if parser is None:
            report.errors.append(f"no parser registered for format {source.format!r}")
            run.reports.append(report)
            return run

        files = self.loader.list_files(source.location)
        report.files_seen = len(files)
        stages = ["load", "parse"]

        if source.pipeline == "tabular":
            units, validation = self._tabular_units(source, parser, files, report)
            stages += ["clean:tabular", "validate:pandera"]
            if validation is not None and not validation.quarantined.empty:
                run.quarantine[source.name] = validation.quarantined
        else:
            units = self._text_units(parser, files, report)
            stages += ["clean:text"]

        report.units_parsed = max(report.units_parsed, len(units))

        if source.sensitivity.contains_pii and self.pseudonymizer is not None:
            units, replaced = self._anonymize(units, source)
            report.entities_anonymized = replaced
            stages.append("anonymize:pseudonymize")
            anonymized = True
        else:
            anonymized = False

        stages.append("normalize")
        documents = [
            to_document(
                unit,
                source=source,
                ingested_at=run.started_at,
                stages=stages,
                anonymized=anonymized,
            )
            for unit in units
        ]
        report.documents_emitted = len(documents)
        run.documents.extend(documents)
        run.reports.append(report)
        logger.info(
            "ingest: %s -> %d documents from %d files",
            source.name, len(documents), len(files),
        )
        return run

    # ------------------------------------------------------------------
    def _tabular_units(
        self, source: CatalogSource, parser, files, report: SourceReport
    ) -> tuple[list[ParsedUnit], ValidationResult | None]:
        records: list[dict] = []
        for ref in files:
            try:
                records.extend(parser.to_records(self.loader.read(ref), ref.path))
            except Exception as exc:  # noqa: BLE001 — one bad file must not lose the source
                report.errors.append(f"{ref.name}: {type(exc).__name__}: {exc}")

        if not records:
            return [], None

        frame = clean_budget_records(pd.DataFrame(records))
        report.divergent_duplicates = list(frame.attrs.get("divergent_budget_ids", []))

        validation = validate_with_policy(frame, BudgetRecord)
        report.records_valid = validation.report.get("valid", 0)
        report.records_quarantined = validation.report.get("quarantined", 0)
        report.records_discarded = validation.report.get("discarded", 0)
        report.units_parsed = len(records)

        units = [
            parser.render_record(record)
            for record in validation.valid.to_dict(orient="records")
        ]
        report.units_after_cleaning = len(units)
        return units, validation

    def _text_units(self, parser, files, report: SourceReport) -> list[ParsedUnit]:
        parsed: list[ParsedUnit] = []
        for ref in files:
            try:
                parsed.extend(parser.parse(self.loader.read(ref), ref.path))
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{ref.name}: {type(exc).__name__}: {exc}")

        report.units_parsed = len(parsed)
        cleaned, cleaning_report = clean_text_units(parsed)
        report.units_after_cleaning = cleaning_report.kept
        return cleaned

    def _anonymize(
        self, units: list[ParsedUnit], source: CatalogSource
    ) -> tuple[list[ParsedUnit], int]:
        out: list[ParsedUnit] = []
        replaced = 0
        for unit in units:
            content, anon_report = self.pseudonymizer.anonymize(
                unit.content, source_name=source.name
            )
            replaced += anon_report.entities_replaced
            # The author field is a name too. Anonymising the body and
            # leaving the speaker in the metadata would leak exactly what
            # the body no longer says.
            author = unit.document_author
            if author:
                author, author_report = self.pseudonymizer.anonymize(
                    author, source_name=source.name
                )
                replaced += author_report.entities_replaced
            out.append(unit.model_copy(update={"content": content, "document_author": author}))
        return out, replaced
