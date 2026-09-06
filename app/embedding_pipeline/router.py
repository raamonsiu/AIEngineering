"""``POST /api/v1/embeddings/ingest``: budgets in, vectors out.

The handler is deliberately thin — chunk, embed, assemble — because both
interesting decisions live one layer down. What it does own is the
failure boundary: anything the OpenAI call raises becomes a 500 carrying
a generic message, while the provider's own wording goes to the logs. A
client has no use for "insufficient_quota on org-xxxx", and an error
string echoed straight back is how account identifiers end up in
somebody else's bug tracker.

The vectors are returned, not stored. That is the shape of this session,
not an oversight: persistence arrives with pgvector in Session 08, and
until then a 1536-float list per chunk travels over HTTP and is forgotten.
Worth knowing before pointing a large corpus at it: 1536 floats serialise
to about 30 KB each, so the 67 chunks of ``data/budgets_sample.json`` come
back as a 2 MB response.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_budgets(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    """Chunk the budgets structurally and embed every chunk.

    Pydantic has already rejected a malformed body with a 422 by the time
    this runs, so the only failure left to handle is the network one.
    """
    chunks = chunker.chunk(request.budgets)

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "embedding_ingest_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:400],
            budgets=len(request.budgets),
            chunks=len(chunks),
        )
        raise HTTPException(
            status_code=500, detail="Embedding provider call failed."
        ) from exc

    # Priced from the chunker's tiktoken counts rather than the API's
    # usage, so stats stay internally consistent: cost is exactly
    # total_tokens x the module price. The embedder logs both numbers per
    # batch, which is where a divergence would show up.
    total_tokens = sum(c.token_count for c in chunks)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
    )

    log.info("embedding_ingest_completed", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
