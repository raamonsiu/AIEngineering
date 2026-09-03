"""Facts-only inspection of the raw data folders.

    uv run python -m app.ingest.census data/corpus
    uv run python -m app.ingest.census data/corpus --against data/data_catalog.yaml

The census is the factual half of the audit, and it is deliberately
separate from the catalog: **facts live in code, opinions live in YAML**.
This module measures file counts, sizes, modification times and formats.
It scores nothing, decides nothing and writes no catalog — those are
human judgements, and mixing them with measurement is how a catalog ends
up asserting things nobody ever checked.

Building the census by hand is the failure mode worth naming: it is how a
catalog comes to claim 28 records when the folder holds 31, and nobody
notices until retrieval starts missing documents that were never indexed.

``--against`` closes that loop. It re-measures every source the catalog
declares and reports the drift, so the catalog can be checked rather than
trusted. That is the check worth running in CI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from app.ingest.catalog import DataCatalog, load_catalog


@dataclass
class FolderFacts:
    """What can be measured from disk, and nothing else."""

    folder: Path
    file_count: int = 0
    total_size_mb: float = 0.0
    latest_modified: datetime | None = None
    formats_detected: set[str] = field(default_factory=set)

    def as_row(self) -> str:
        stamp = self.latest_modified.date().isoformat() if self.latest_modified else "—"
        formats = ",".join(sorted(self.formats_detected)) or "—"
        return (
            f"  {self.folder.name:<24}{self.file_count:>5} ficheros"
            f"{self.total_size_mb:>9.3f} MB   ult={stamp:<12} formatos={formats}"
        )


def inspect_folder(folder: Path) -> FolderFacts:
    facts = FolderFacts(folder=folder)
    if not folder.exists():
        return facts
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        facts.file_count += 1
        facts.total_size_mb += stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        if facts.latest_modified is None or modified > facts.latest_modified:
            facts.latest_modified = modified
        if path.suffix:
            facts.formats_detected.add(path.suffix.lower().lstrip("."))
    facts.total_size_mb = round(facts.total_size_mb, 3)
    return facts


def inspect_root(root: Path) -> list[FolderFacts]:
    """Inspect every immediate subfolder of ``root``, plus loose files."""
    if not root.exists():
        return []
    results = [inspect_folder(child) for child in sorted(root.iterdir()) if child.is_dir()]
    loose = [p for p in sorted(root.iterdir()) if p.is_file()]
    if loose:
        bag = FolderFacts(folder=root)
        for path in loose:
            stat = path.stat()
            bag.file_count += 1
            bag.total_size_mb += stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if bag.latest_modified is None or modified > bag.latest_modified:
                bag.latest_modified = modified
            if path.suffix:
                bag.formats_detected.add(path.suffix.lower().lstrip("."))
        results.append(bag)
    return results


@dataclass
class Drift:
    """One discrepancy between what the catalog claims and what is on disk."""

    source_name: str
    field: str
    declared: str
    measured: str

    def as_row(self) -> str:
        return f"  {self.source_name:<24}{self.field:<22}declara={self.declared:<14}mide={self.measured}"


def check_against_catalog(
    catalog: DataCatalog, corpus_root: Path, *, audited_on: date | None = None
) -> list[Drift]:
    """Re-measure every declared source and report where the catalog lies.

    Size is compared with a tolerance; counts and dates are not. A record
    count that is off by one means a file was added or lost without anyone
    updating the audit, and that is exactly the drift worth catching.
    """
    reference = audited_on or catalog.last_audited
    drifts: list[Drift] = []

    for source in catalog.sources:
        facts = inspect_folder(source.local_path(corpus_root))

        if facts.file_count != source.volume.records:
            drifts.append(Drift(source.name, "volume.records",
                                str(source.volume.records), str(facts.file_count)))

        declared_mb = source.volume.size_mb
        if declared_mb and abs(facts.total_size_mb - declared_mb) > max(0.005, declared_mb * 0.10):
            drifts.append(Drift(source.name, "volume.size_mb",
                                f"{declared_mb:.3f}", f"{facts.total_size_mb:.3f}"))

        if facts.latest_modified is not None:
            measured_date = facts.latest_modified.date()
            if measured_date != source.refresh.observed_last_update:
                drifts.append(Drift(source.name, "refresh.observed_last_update",
                                    str(source.refresh.observed_last_update), str(measured_date)))
            measured_lag = (reference - measured_date).days
            if abs(measured_lag - source.refresh.observed_lag_days) > 1:
                drifts.append(Drift(source.name, "refresh.observed_lag_days",
                                    str(source.refresh.observed_lag_days), str(measured_lag)))

        if facts.formats_detected and source.format not in facts.formats_detected:
            drifts.append(Drift(source.name, "format", source.format,
                                ",".join(sorted(facts.formats_detected))))

    return drifts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/corpus")
    parser.add_argument("--against", default=None, help="catalog to verify against")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} no existe", file=sys.stderr)
        return 2

    print(f"Censo de {root.resolve()}\n")
    facts = inspect_root(root)
    if not facts:
        print("  (sin ficheros)")
        return 0
    for entry in facts:
        print(entry.as_row())
    print(
        f"\n  {'TOTAL':<24}{sum(f.file_count for f in facts):>5} ficheros"
        f"{sum(f.total_size_mb for f in facts):>9.3f} MB"
    )

    if not args.against:
        return 0

    catalog = load_catalog(Path(args.against))
    drifts = check_against_catalog(catalog, root)
    print(f"\nContraste con {args.against} ({len(catalog.sources)} fuentes declaradas)\n")
    if not drifts:
        print("  sin desviaciones: el catálogo describe lo que hay en disco")
        return 0
    for drift in drifts:
        print(drift.as_row())
    print(f"\n  {len(drifts)} desviacion(es). El catálogo necesita una re-auditoría.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
