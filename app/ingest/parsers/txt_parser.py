"""Meeting-transcript parser.

TXT looks like the easiest format and is the one that most rewards not
treating it as a bag of text. These transcripts come in two eras:

- 2024 onwards, from the automated service: ``[hh:mm:ss] Nombre: texto``
- before 2024, hand-kept: free-running paragraphs, no speaker tags

Flattening both into one blob throws away the single most valuable signal
a meeting carries — who said what. A client stating a budget and a
salesperson speculating about one are different facts, and after
flattening they are the same string. So the parser detects the era and
emits one unit per *turn* where turns exist, carrying speaker and
timestamp as metadata; where they do not, it falls back to paragraphs and
says so, which is itself a quality signal worth recording.
"""

from __future__ import annotations

import re

from app.ingest.models import ParsedUnit
from app.ingest.parsers.base import Parser

# [hh:mm:ss] Speaker Name: utterance
_TAGGED_TURN = re.compile(
    r"^\s*\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<speaker>[^:]{1,80}?)\s*:\s*(?P<text>.+)$"
)
# Minimum share of lines that must match before we trust the tagged format.
# A handful of accidental "word: text" lines in a legacy transcript should
# not promote it to the tagged era.
_TAGGED_THRESHOLD = 0.5


class TranscriptTxtParser:
    supported_formats = {"txt"}

    def _decode(self, content: bytes) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")

    def parse(self, content: bytes, source_hint: str) -> list[ParsedUnit]:
        text = self._decode(content)
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        matches = [_TAGGED_TURN.match(line) for line in lines]
        tagged_ratio = sum(1 for m in matches if m) / len(lines)
        doc_id = source_hint.rsplit("/", 1)[-1]

        if tagged_ratio >= _TAGGED_THRESHOLD:
            return self._parse_tagged(matches, lines, doc_id)
        return self._parse_untagged(text, doc_id)

    @staticmethod
    def _parse_tagged(matches, lines, doc_id: str) -> list[ParsedUnit]:
        units: list[ParsedUnit] = []
        for index, (match, line) in enumerate(zip(matches, lines), start=1):
            if match is None:
                # An untagged line inside a tagged transcript is kept, not
                # dropped: losing content to preserve a format assumption
                # is the wrong trade.
                units.append(
                    ParsedUnit(
                        content=line.strip(),
                        document_id=doc_id,
                        section_title=f"turno {index}",
                        extra={"format_era": "tagged", "speaker": None, "untagged_line": True},
                    )
                )
                continue
            units.append(
                ParsedUnit(
                    content=match.group("text").strip(),
                    document_id=doc_id,
                    section_title=f"turno {index}",
                    document_author=match.group("speaker").strip(),
                    extra={
                        "format_era": "tagged",
                        "speaker": match.group("speaker").strip(),
                        "timestamp": match.group("ts"),
                    },
                )
            )
        return units

    @staticmethod
    def _parse_untagged(text: str, doc_id: str) -> list[ParsedUnit]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [
            ParsedUnit(
                content=paragraph,
                document_id=doc_id,
                section_title=f"parrafo {index}",
                extra={
                    "format_era": "legacy_untagged",
                    "speaker": None,
                    # Recorded so the catalog's consistency score has
                    # evidence behind it, and so a later reviewer can find
                    # every unit whose attribution is unknown.
                    "speaker_attribution": "unavailable",
                },
            )
            for index, paragraph in enumerate(paragraphs, start=1)
        ]


_: Parser = TranscriptTxtParser()
