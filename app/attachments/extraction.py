"""Online-path policy for user-uploaded attachments.

Extraction itself lives in ``app.ingest.parsers.extraction`` — there is one
place in this project where bytes become text, and this is not it. What
belongs here is what only the *live conversation* path cares about: the
per-attachment word cap, and rendering accepted attachments into the block
the prompt expects.

The cap is deliberately not applied by the ingest pipeline. Here the
problem is context budget and a user waiting; offline, neither applies.
"""

from __future__ import annotations

from dataclasses import replace

from app.ingest.parsers.extraction import ExtractionResult

def check_length(text: str, max_words: int) -> tuple[bool, str | None]:
    """Abstracted, non-technical size check: word count, not tokens, a
    number a non-technical user can actually reason about. Deliberately
    reports both the absolute count and the multiple over the limit, so the
    user can tell "barely over" from "an order of magnitude too big"."""
    word_count = len(text.split())
    if word_count <= max_words:
        return True, None
    ratio = word_count / max_words
    return False, (
        f"This document has about {word_count:,} words; the maximum allowed "
        f"is {max_words:,} (~{ratio:.1f}x over the limit)."
    )


def enforce_length_limit(result: ExtractionResult, max_words: int) -> ExtractionResult:
    """Downgrade an otherwise-successful extraction to a reported failure
    when its text is too long. This runs AFTER extraction succeeds, and is
    deliberately not folded into the multimodal fallback: the problem here
    is size/cost, not readability, so re-sending the same content as a raw
    file to the LLM would make the cost problem worse, not solve it."""
    if not result.ok:
        return result
    ok, note = check_length(result.text, max_words)
    if ok:
        return result
    return replace(result, ok=False, text="", method="too_long", note=note)


def format_attachments_block(results: list[ExtractionResult]) -> str:
    """Render successfully-extracted attachments as the delimited blocks the
    prompt expects. Failed ones are omitted here, the caller decides whether
    to note the failure in text or retry via the LLM's multimodal fallback."""
    blocks = [
        f"<attachment filename='{result.filename}'>\n{result.text}\n</attachment>"
        for result in results
        if result.ok
    ]
    return "\n\n".join(blocks)
