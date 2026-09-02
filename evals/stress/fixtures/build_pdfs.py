"""Generate the synthetic attachment corpus for the attachment sweep.

Run directly, or let ``evals.stress.run`` regenerate it:

    uv run python -m evals.stress.fixtures.build_pdfs

Design decisions
----------------
**PyMuPDF, not reportlab or fpdf2.** Both would work and the brief allows
either, but PyMuPDF is already a dependency of this service (it is the
second-chance extractor in ``app/attachments/extraction.py``). Adding a PDF
library to generate fixtures for a measurement run would put a new package
in the project to produce data that never ships.

**Calibrated on extracted-text size, not file size.** "A 50 KB attachment"
could mean 50 KB of PDF on disk or 50 KB of text in the prompt, and for a
context-window study only the second one means anything: PDF file size
swings with fonts, compression and embedded metadata, none of which reach
the model. So each fixture is built, run through the service's real
extraction path, measured, and resized until the *extracted* text lands on
target. That loop also means the numbers in the report describe what the
LLM actually saw.

**Three markers, at head, middle and tail.** A single marker would only
answer "did the attachment reach the model". Three answer the more useful
question: *where* in a long attachment does content stop surviving. The
head marker is the one a model is most likely to retain and the middle one
the least, so the gap between them measures position effects directly
rather than assuming them. They are written as plausible project facts
rather than obvious sentinels, so the model has no reason to treat them as
special.

**Deterministic, and not committed.** Same input, same bytes, every run —
the filler is drawn from a fixed word list with a fixed seed, so the corpus
is reproducible from this script and does not need to live in git.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pymupdf

FIXTURES_DIR = Path(__file__).resolve().parent

# The sizes the brief asks for, in KB of EXTRACTED TEXT. 0 KB is the
# no-attachment baseline and needs no file.
DEFAULT_SIZES_KB: tuple[int, ...] = (5, 20, 50, 100)

# Fixed seed: the corpus must be byte-identical across runs, or two stress
# runs are not comparable.
SEED = 20260604


@dataclass(frozen=True)
class Marker:
    """A distinctive fact planted at a known position in the document."""

    key: str
    position: str  # "head" | "middle" | "tail"
    sentence: str
    needles: tuple[str, ...]  # surface forms that count as recalled


MARKERS: tuple[Marker, ...] = (
    Marker(
        key="ref_code",
        position="head",
        sentence=(
            "The client reference code for this engagement is ORION-7731 and must be "
            "quoted on every deliverable."
        ),
        needles=("ORION-7731", "ORION 7731"),
    ),
    Marker(
        key="pilot_site",
        position="middle",
        sentence=(
            "The pilot deployment will run at the Zaragoza depot before any wider rollout "
            "is considered."
        ),
        needles=("Zaragoza",),
    ),
    Marker(
        key="deadline",
        position="tail",
        sentence=(
            "The hard deadline for phase one acceptance is 14 March 2027, fixed by the "
            "client's regulatory filing."
        ),
        needles=("14 March 2027", "March 2027", "2027-03-14", "14/03/2027"),
    ),
)

# Domain-flavoured filler. Deliberately on-topic: lorem ipsum would be
# trivially ignorable by the model, which would flatter the recall numbers.
_VOCAB = (
    "requirement stakeholder deployment migration integration workflow scheduling "
    "maintenance inventory procurement compliance reporting dashboard interface "
    "provisioning escalation remediation throughput availability redundancy "
    "onboarding specification acceptance validation rollout governance baseline "
    "dependency mitigation contingency allocation utilisation benchmark audit "
    "retention replication orchestration observability resilience"
).split()

_SENTENCE_STARTS = (
    "The supplier confirms that",
    "Section four notes that",
    "Operations have requested that",
    "The steering committee agreed that",
    "Current practice assumes that",
    "The technical annex states that",
)


def _filler_sentence(rng: random.Random) -> str:
    start = rng.choice(_SENTENCE_STARTS)
    words = [rng.choice(_VOCAB) for _ in range(rng.randint(8, 18))]
    return f"{start} {' '.join(words)}."


def _build_text(target_chars: int, rng: random.Random) -> str:
    """Filler with the three markers planted at head, middle and tail."""
    head = next(m for m in MARKERS if m.position == "head")
    middle = next(m for m in MARKERS if m.position == "middle")
    tail = next(m for m in MARKERS if m.position == "tail")

    body: list[str] = [
        "PROJECT REQUIREMENTS BRIEF",
        "",
        head.sentence,
        "",
    ]
    while len(" ".join(body)) < target_chars:
        body.append(_filler_sentence(rng))
        # Plant the middle marker once, as close to the centre as the
        # growing document allows.
        if middle.sentence not in body and len(" ".join(body)) >= target_chars // 2:
            body.append(middle.sentence)

    if middle.sentence not in body:
        body.insert(len(body) // 2, middle.sentence)
    body.extend(["", tail.sentence])
    return "\n".join(body)


def _render_pdf(text: str, *, words_per_page: int = 420) -> bytes:
    """Lay the text out over as many A4 pages as it needs.

    Pages are filled by word count rather than by asking PyMuPDF where the
    text overflowed: ``insert_textbox`` reports *that* the text did not fit,
    not *where* it was cut, so a word-count split is the only way to keep
    the document's content exactly equal to the text we generated. The
    return code is still checked, and the page is re-laid at a smaller font
    if a chunk overflows, so no content is ever silently dropped.
    """
    words = text.split(" ")
    doc = pymupdf.open()
    rect = pymupdf.Rect(56, 56, 539, 786)  # A4 with ~20 mm margins

    for start in range(0, len(words), words_per_page):
        chunk = " ".join(words[start : start + words_per_page])
        page = doc.new_page()
        for fontsize in (9, 8, 7, 6):
            leftover = page.insert_textbox(rect, chunk, fontsize=fontsize, fontname="helv")
            if leftover >= 0:
                break
        else:  # pragma: no cover — only if words_per_page is set absurdly high
            raise RuntimeError(
                f"chunk of {words_per_page} words will not fit on a page even at 6pt"
            )

    data = doc.tobytes()
    doc.close()
    return data


def _extracted_chars(raw: bytes, filename: str) -> int:
    """Measure with the service's own extractor, so the calibration target
    is the text the prompt will really carry, headers and all."""
    from app.attachments import extract_attachment

    result = extract_attachment(filename, raw)
    if not result.ok:
        raise RuntimeError(f"generated PDF failed extraction: {result.note}")
    return len(result.text)


def build_pdf(target_kb: int, *, tolerance: float = 0.03, max_rounds: int = 12) -> bytes:
    """Build one fixture whose EXTRACTED text is within ``tolerance`` of
    ``target_kb``.

    Closed loop rather than open: generate, extract, compare, rescale. PDF
    text extraction inserts page headers and normalises whitespace, so the
    text that goes in is never exactly the text that comes out, and a
    one-shot generator would be off by a few percent in a way that varies
    with document length.
    """
    filename = f"attach_{target_kb}kb.pdf"
    target_chars = target_kb * 1024
    generate_chars = target_chars
    raw = b""

    for _ in range(max_rounds):
        rng = random.Random(SEED + target_kb)
        raw = _render_pdf(_build_text(generate_chars, rng))
        actual = _extracted_chars(raw, filename)
        drift = (actual - target_chars) / target_chars
        if abs(drift) <= tolerance:
            return raw
        # Damped correction: the relationship is close to linear but the
        # per-page headers make it slightly step-shaped, and an undamped
        # step oscillates around the target instead of settling.
        generate_chars = max(256, int(generate_chars * (1 - drift * 0.8)))

    return raw  # best effort; the manifest records what was actually achieved


def build_all(sizes_kb: tuple[int, ...] = DEFAULT_SIZES_KB) -> list[dict]:
    from app.attachments import extract_attachment

    manifest: list[dict] = []
    for kb in sizes_kb:
        path = FIXTURES_DIR / f"attach_{kb}kb.pdf"
        raw = build_pdf(kb)
        path.write_bytes(raw)
        extracted = extract_attachment(path.name, raw)
        manifest.append(
            {
                "target_kb": kb,
                "path": path,
                "file_bytes": len(raw),
                "extracted_chars": len(extracted.text),
                "extracted_words": len(extracted.text.split()),
                "pages": pymupdf.open(stream=raw, filetype="pdf").page_count,
            }
        )
    return manifest


def ensure_corpus(sizes_kb: tuple[int, ...] = DEFAULT_SIZES_KB) -> dict[int, Path]:
    """Return ``{size_kb: path}``, building anything missing.

    0 KB maps to no path: the baseline is the *absence* of an attachment,
    not an empty file, because an empty PDF would fail extraction and be
    reported as a failed attachment — a different code path from the one
    the baseline is supposed to represent.
    """
    paths: dict[int, Path] = {}
    missing: list[int] = []
    for kb in sizes_kb:
        if kb == 0:
            continue
        path = FIXTURES_DIR / f"attach_{kb}kb.pdf"
        paths[kb] = path
        if not path.exists():
            missing.append(kb)
    if missing:
        build_all(tuple(missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES_KB),
        help="comma-separated target sizes in KB of extracted text",
    )
    args = parser.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(",") if int(s) > 0)

    print(f"{'target':>8} {'pages':>6} {'file KB':>9} {'text KB':>9} {'words':>8} {'drift':>7}")
    for entry in build_all(sizes):
        text_kb = entry["extracted_chars"] / 1024
        drift = (text_kb - entry["target_kb"]) / entry["target_kb"]
        print(
            f"{entry['target_kb']:>6} KB {entry['pages']:>6} "
            f"{entry['file_bytes'] / 1024:>9.1f} {text_kb:>9.1f} "
            f"{entry['extracted_words']:>8} {drift:>+6.1%}"
        )


if __name__ == "__main__":
    main()
