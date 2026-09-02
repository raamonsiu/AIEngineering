"""Filesystem loader: how we reach the bytes, never what is inside them.

The layer exists so a format is parsed once regardless of where it lives.
A PDF in a local folder, in Drive and in an S3 bucket is the same PDF; if
the parser knew about storage, there would be three PDF parsers. Adding a
Drive or S3 loader later means implementing ``list_files``/``read`` and
changing nothing downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FileRef:
    """A file the loader can reach, plus the facts the audit needs.

    ``modified_at`` and ``size_bytes`` come from the storage layer rather
    than the document: they are what makes "declared cadence vs observed
    cadence" checkable, and no parser can recover them once the bytes are
    detached from their file.
    """

    path: str
    name: str
    size_bytes: int
    modified_at: datetime
    suffix: str


class Loader(Protocol):
    def list_files(self, location: str) -> list[FileRef]: ...
    def read(self, ref: FileRef) -> bytes: ...


class FilesystemLoader:
    """Reads from a local directory tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list_files(self, location: str) -> list[FileRef]:
        """List a source's files, sorted by name.

        Sorted deliberately: ingestion order decides which of two
        divergent duplicates the pipeline sees first, and an order that
        depends on the filesystem's directory iteration is an order the
        team does not control. Deterministic order makes a re-run
        reproducible and a bug reproducible with it.
        """
        base = self.root / location.partition("://")[2].strip("/") if "://" in location else Path(location)
        if not base.exists():
            return []
        return [
            FileRef(
                path=str(path),
                name=path.name,
                size_bytes=path.stat().st_size,
                modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                suffix=path.suffix.lower().lstrip("."),
            )
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]

    def read(self, ref: FileRef) -> bytes:
        return Path(ref.path).read_bytes()
