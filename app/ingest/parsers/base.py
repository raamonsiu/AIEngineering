"""The parser contract.

``Parser`` is a ``Protocol``, not an abstract base class: structural typing
means anything with the right ``parse`` signature qualifies, so a parser
can be a plain class, a dataclass or a closure without inheriting from
this module. It also keeps test doubles trivial — a fake parser is six
lines and imports nothing.

Parsers return ``ParsedUnit``, not ``Document``. See ``app/ingest/models.py``
for why that gap is deliberate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.ingest.models import ParsedUnit


@runtime_checkable
class Parser(Protocol):
    """Turns raw bytes of one known format into addressable text units."""

    supported_formats: set[str]

    def parse(self, content: bytes, source_hint: str) -> list[ParsedUnit]: ...


@runtime_checkable
class TabularParser(Parser, Protocol):
    """A parser whose format is records, not prose.

    These take a second route through the pipeline: their intermediate
    representation is a list of dicts that becomes a DataFrame, gets
    cleaned and validated against a Pandera schema, and only then is
    rendered to text. ``parse`` still works (it renders straight through),
    but the orchestrator prefers ``to_records`` for included sources so
    the validation layer gets its say.
    """

    def to_records(self, content: bytes, source_hint: str) -> list[dict]: ...

    def render_record(self, record: dict) -> ParsedUnit: ...
