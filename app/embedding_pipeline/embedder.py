"""The only place in this module that talks to OpenAI.

Three things here are not incidental:

**Batching.** ``embeddings.create`` accepts a list, and the round trip
dominates the cost of embedding short texts. Sixty-seven serialised calls
to embed sixty-seven chunks would spend most of a minute waiting on the
network to do a few hundred milliseconds of work. One call per hundred
chunks turns that into a single round trip.

**Positional reassembly.** The API returns an ``index`` per embedding.
The code sorts on it instead of trusting arrival order, because a vector
silently attached to the wrong chunk is the hardest bug in this module to
notice: nothing errors, retrieval just quietly returns the wrong budget.

**One retry policy, not two.** The OpenAI SDK retries 429s on its own.
Layering the explicit backoff this module needs on top of that would make
the real behaviour the *product* of both policies and the logs a lie
about how many attempts happened, so the SDK's retries are turned off
here and the policy below is the whole policy.
"""

from __future__ import annotations

import time

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

EMBEDDING_MODEL = "text-embedding-3-small"

# The model's native output width. Not reduced: text-embedding-3-small
# supports truncating the vector (Matryoshka), which trades recall for
# storage, and that trade has no basis until there is an index to measure
# it against.
EMBEDDING_DIMENSIONS = 1536

# PRICE, AS OF 2026-10: $0.02 per million input tokens for
# text-embedding-3-small. This WILL go stale — it is a vendor price, not a
# constant of nature. The cost it feeds is labelled "estimated" everywhere
# it surfaces for that reason. The chat-model equivalent lives in
# ``app/constants.py``; embeddings have no output tokens, so they do not
# fit that table's input/output shape.
EMBEDDING_PRICE_USD_PER_MILLION_TOKENS = 0.02

# Large enough that one request covers a typical ingest, small enough to
# stay well under the API's per-request token ceiling.
DEFAULT_BATCH_SIZE = 100

# Waits before the 1st, 2nd and 3rd retry. Four attempts in total, and at
# most seven seconds of sleeping before the error is allowed to propagate.
RATE_LIMIT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


def estimate_cost_usd(total_tokens: int) -> float:
    """Token count to dollars. Separate from the client so it can be called
    without one — the router prices a run from the chunker's counts."""
    return total_tokens * EMBEDDING_PRICE_USD_PER_MILLION_TOKENS / 1_000_000


class OpenAIEmbedder:
    """Turns text into vectors, in batches, with a retry policy."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str = EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # ``max_retries=0``: see the module docstring. The client is
        # injected rather than built here so this class stays testable
        # against a double and shares the process-wide singleton in
        # production (``app/dependencies.py``).
        self._client = client.with_options(max_retries=0)
        self._model = model
        self._batch_size = batch_size

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Used by ``scripts/compare.py``, where
        there is no chunk to attach the result to."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks in batches, returning them in the order given."""
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._create(
                [c.text for c in batch],
                batch_index=start // self._batch_size,
                estimated_tokens=sum(c.token_count for c in batch),
            )
            embedded.extend(
                EmbeddedChunk(**c.model_dump(), embedding=v)
                for c, v in zip(batch, vectors, strict=True)
            )
        return embedded

    def _create(
        self,
        texts: list[str],
        *,
        batch_index: int = 0,
        estimated_tokens: int | None = None,
    ) -> list[list[float]]:
        """One API call, with the rate-limit policy around it.

        Every exception other than ``RateLimitError`` propagates untouched:
        an invalid key or an oversized input will not get better by waiting,
        and swallowing them here would turn a clear 401 into a mysterious
        delay followed by the same 401.
        """
        started = time.perf_counter()
        for attempt, wait in enumerate((*RATE_LIMIT_BACKOFF_SECONDS, None)):
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                break
            except RateLimitError:
                if wait is None:
                    log.error(
                        "embedding_rate_limited_giving_up",
                        batch_index=batch_index,
                        attempts=attempt + 1,
                    )
                    raise
                log.warning(
                    "embedding_rate_limited_retrying",
                    batch_index=batch_index,
                    attempt=attempt + 1,
                    sleeping_s=wait,
                )
                time.sleep(wait)

        latency_ms = int((time.perf_counter() - started) * 1000)
        # Sorted by the API's own index rather than trusting arrival order.
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

        if any(len(v) != EMBEDDING_DIMENSIONS for v in vectors):
            raise ValueError(
                f"{self._model} returned a vector that is not "
                f"{EMBEDDING_DIMENSIONS}-dimensional; the model or its "
                f"dimensions parameter changed under us"
            )

        billed_tokens = response.usage.prompt_tokens
        log.info(
            "embedding_batch_completed",
            batch_index=batch_index,
            chunks=len(texts),
            # Both numbers, because they come from different places: the
            # estimate is tiktoken's local count (what the chunker reported
            # and what the cost is based on), the billed one is the API's.
            # A gap between them means the tokenizer is wrong for the model.
            tokens_estimated=estimated_tokens,
            tokens_billed=billed_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )
        return vectors
