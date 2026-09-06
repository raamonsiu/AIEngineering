"""Budgets to vectors: chunk structurally, embed in batches, return.

    Budget[] -> JSONStructuralChunker -> Chunk[] -> OpenAIEmbedder -> EmbeddedChunk[]

The two stages answer different questions and are kept apart for that
reason. The chunker decides *what a unit of meaning is* — here, one budget
component, because the JSON already drew that line — and is pure, offline
and free to run. The embedder decides *how a unit becomes a vector* and is
the only part that costs money and can fail.

This is the first half of retrieval. The second half (a vector index,
similarity search, the ``/query`` endpoint that still answers 501) needs
somewhere to put these vectors, which is Session 08's job.
"""

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    ClientMetadata,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    IngestStats,
)

__all__ = [
    "Budget",
    "BudgetComponent",
    "Chunk",
    "ClientMetadata",
    "EmbeddedChunk",
    "IngestRequest",
    "IngestResponse",
    "IngestStats",
    "JSONStructuralChunker",
    "OpenAIEmbedder",
    "estimate_cost_usd",
]
