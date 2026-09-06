#!/usr/bin/env python
"""Embed two texts and print the cosine similarity between them.

A sanity check, not a benchmark. It answers one question — does this
embedding model put texts that mean similar things near each other — and
it answers it cheaply enough to run before trusting any retrieval built on
top. Two short texts cost around four hundredths of a cent.

    uv run python scripts/compare.py \\
      --text-a "OAuth 2.0 authentication backend for fintech" \\
      --text-b "JWT-based authorization service for banking app"

Inside the container:

    docker compose exec cag-estimator python scripts/compare.py --text-a ... --text-b ...

Run it from the repository root either way: settings are read from ``.env``
relative to the working directory, and ``app`` has to be importable.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Make ``app`` importable when the script is run directly from the repo
# root (``python scripts/compare.py``), where sys.path[0] is scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product over the product of the norms, by hand.

    ``math.fsum`` and ``math.hypot`` rather than a plain loop and a
    ``sqrt(sum(x*x))``: both are stdlib, and both avoid the rounding error
    that accumulates over 1536 terms of a float sum. Pulling in numpy for
    three lines of arithmetic would be the wrong trade.

    Note for interpreting the result: OpenAI returns unit-length vectors,
    so both norms are 1.0 and this reduces to the dot product. The division
    is kept because the function should not silently depend on that.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")

    norm_a, norm_b = math.hypot(*a), math.hypot(*b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    return dot / (norm_a * norm_b)


def _configure_logging(verbose: bool) -> None:
    """Keep stdout clean.

    The embedder logs every batch through structlog, which is what you
    want in the API and noise in a CLI whose output is three lines. The
    logs go to stderr and stay below WARNING unless ``--verbose`` is
    passed, so piping stdout somewhere still works.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(10 if verbose else 30),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cosine similarity between the embeddings of two texts.",
    )
    parser.add_argument("--text-a", required=True, help="First text.")
    parser.add_argument("--text-b", required=True, help="Second text.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the embedder's per-batch log (tokens, latency) on stderr.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        # A clean message, not a traceback: the fix is a line in .env, and
        # a stack trace would bury that under twenty frames.
        print("OPENAI_API_KEY is not set (check .env in the repo root).", file=sys.stderr)
        return 2

    from openai import OpenAI

    embedder = OpenAIEmbedder(OpenAI(api_key=settings.OPENAI_API_KEY))

    try:
        vector_a = embedder.embed_one(args.text_a)
        vector_b = embedder.embed_one(args.text_b)
    except Exception as exc:  # noqa: BLE001
        print(f"Embedding call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {cosine_similarity(vector_a, vector_b):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
