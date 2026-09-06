"""Tests for the chunking and embedding stages.

Nothing here touches the network. The chunker is pure, and the embedder is
exercised against a double that counts calls, so the batching and the
rate-limit policy are asserted rather than assumed — those are precisely
the two behaviours that are expensive to discover in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from openai import RateLimitError

from app.embedding_pipeline.chunker import (
    MAX_TOKENS_PER_CHUNK,
    REQUIRED_METADATA_KEYS,
    JSONStructuralChunker,
)
from app.embedding_pipeline.embedder import (
    EMBEDDING_DIMENSIONS,
    OpenAIEmbedder,
    estimate_cost_usd,
)
from app.embedding_pipeline.schemas import Budget, Chunk, IngestRequest

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"


def _budget(**overrides) -> Budget:
    base = {
        "budget_id": "BUD-2024-001",
        "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
        "project_summary": "Mobile banking API with PSD2 compliance",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "total_estimated_hours": 200,
        "components": [
            {
                "component_id": "AUTH-001",
                "name": "OAuth 2.0 authentication backend",
                "description": "Authorization code and refresh token flows with JWT sessions.",
                "tech_stack": ["ruby_on_rails", "postgresql"],
                "estimated_hours": 120,
                "complexity": "high",
                "dependencies": [],
            }
        ],
    }
    return Budget.model_validate({**base, **overrides})


# ----------------------------------------------------------------------
# Chunker
# ----------------------------------------------------------------------
def test_one_component_is_one_chunk() -> None:
    budget = _budget(
        components=[
            {
                "component_id": f"C-00{i}",
                "name": f"Component {i}",
                "description": "Does a thing.",
                "tech_stack": ["go"],
                "estimated_hours": 10,
                "complexity": "low",
                "dependencies": [],
            }
            for i in range(1, 4)
        ]
    )
    chunks = JSONStructuralChunker().chunk([budget])
    assert [c.chunk_id for c in chunks] == [
        "BUD-2024-001::C-001", "BUD-2024-001::C-002", "BUD-2024-001::C-003"
    ]


def test_chunk_text_carries_the_parent_context() -> None:
    """Without the header a component is unattributable: the same
    "authentication backend" could belong to a bank or a bike shop, and the
    vectors would be identical."""
    text = JSONStructuralChunker().chunk([_budget()])[0].text
    assert "Mobile banking API with PSD2 compliance" in text
    assert "Client sector: finance" in text
    assert "Main tech: ruby_on_rails" in text
    # ...and the component's own detail, which is what distinguishes it
    # from every other component of the same budget.
    assert "Authorization code and refresh token flows" in text


def test_filterable_fields_stay_out_of_the_embedded_text() -> None:
    """Metadata exists so that "2024" is a constraint, not four tokens of
    noise in a 1536-dimensional vector."""
    chunk = JSONStructuralChunker().chunk([_budget()])[0]
    assert REQUIRED_METADATA_KEYS <= set(chunk.metadata)
    assert chunk.metadata["budget_id"] == "BUD-2024-001"
    assert chunk.metadata["client_sector"] == "finance"
    assert "BUD-2024-001" not in chunk.text


def test_integral_hours_render_without_a_trailing_zero() -> None:
    whole = JSONStructuralChunker().chunk([_budget()])[0]
    assert "Estimated hours: 120" in whole.text and "120.0" not in whole.text


def test_token_count_is_reported_and_below_the_model_limit() -> None:
    chunks = JSONStructuralChunker().chunk([_budget()])
    assert 0 < chunks[0].token_count <= MAX_TOKENS_PER_CHUNK


def test_sample_corpus_chunks_cleanly() -> None:
    """The committed sample is part of the contract: if it stops validating
    or starts colliding on ids, the README's instructions are broken."""
    request = IngestRequest.model_validate(json.loads(SAMPLE_PATH.read_text(encoding="utf-8")))
    chunks = JSONStructuralChunker().chunk(request.budgets)

    assert len(chunks) == sum(len(b.components) for b in request.budgets)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.token_count <= MAX_TOKENS_PER_CHUNK for c in chunks)


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------
def test_sector_aliases_fold_onto_the_canonical_value() -> None:
    budget = _budget(client_metadata={"name": "X", "sector": "FinTech", "country": "ES"})
    assert budget.client_metadata.sector == "finance"


def test_unknown_sector_is_rejected_rather_than_passed_through() -> None:
    """A sector nobody filters on is worse than an error: the query returns
    zero results and looks like a retrieval problem."""
    with pytest.raises(ValueError):
        _budget(client_metadata={"name": "X", "sector": "aerospace", "country": "ES"})


def test_ids_containing_the_chunk_separator_are_rejected() -> None:
    """``{budget_id}::{component_id}`` stops being parseable if either half
    contains the separator."""
    with pytest.raises(ValueError, match="must not contain"):
        _budget(budget_id="BUD::001")


# ----------------------------------------------------------------------
# Embedder
# ----------------------------------------------------------------------
def _rate_limit_error() -> RateLimitError:
    """The SDK's exception needs a real response to build from."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


class _FakeEmbeddings:
    def __init__(self, owner: "_FakeOpenAI") -> None:
        self._owner = owner

    def create(self, *, model: str, input: list[str]):
        self._owner.batch_sizes.append(len(input))
        if self._owner.fail_times > 0:
            self._owner.fail_times -= 1
            raise _rate_limit_error()

        # Returned deliberately out of order, with the API's own index, to
        # prove the embedder reassembles positionally instead of trusting
        # the order it happens to receive.
        data = [
            type("Item", (), {"index": i, "embedding": [float(i)] * EMBEDDING_DIMENSIONS})()
            for i in range(len(input))
        ]
        usage = type("Usage", (), {"prompt_tokens": sum(len(t) for t in input)})()
        return type("Response", (), {"data": list(reversed(data)), "usage": usage})()


class _FakeOpenAI:
    def __init__(self, fail_times: int = 0) -> None:
        self.batch_sizes: list[int] = []
        self.fail_times = fail_times
        self.embeddings = _FakeEmbeddings(self)

    def with_options(self, **_kwargs) -> "_FakeOpenAI":
        return self


def _chunks(count: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"B::C{i}", text=f"text {i}", metadata={}, token_count=2)
        for i in range(count)
    ]


def test_embed_many_batches_instead_of_one_call_per_chunk() -> None:
    client = _FakeOpenAI()
    embedded = OpenAIEmbedder(client, batch_size=100).embed_many(_chunks(250))

    assert client.batch_sizes == [100, 100, 50]
    assert len(embedded) == 250


def test_vectors_are_matched_to_chunks_by_index_not_arrival_order() -> None:
    """The failure this guards against is silent: a misaligned vector makes
    retrieval return the wrong budget without anything erroring."""
    embedded = OpenAIEmbedder(_FakeOpenAI(), batch_size=10).embed_many(_chunks(3))

    assert [c.chunk_id for c in embedded] == ["B::C0", "B::C1", "B::C2"]
    assert [c.embedding[0] for c in embedded] == [0.0, 1.0, 2.0]


def test_chunk_fields_survive_embedding() -> None:
    chunk = _chunks(1)[0]
    embedded = OpenAIEmbedder(_FakeOpenAI()).embed_many([chunk])[0]

    assert (embedded.chunk_id, embedded.text, embedded.token_count) == (
        chunk.chunk_id, chunk.text, chunk.token_count
    )
    assert len(embedded.embedding) == EMBEDDING_DIMENSIONS


def test_rate_limit_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("app.embedding_pipeline.embedder.time.sleep", slept.append)

    client = _FakeOpenAI(fail_times=2)
    embedded = OpenAIEmbedder(client).embed_many(_chunks(1))

    assert slept == [1.0, 2.0]
    assert len(embedded) == 1


def test_rate_limit_propagates_once_the_retries_are_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four attempts, then the caller finds out. Retrying forever would turn
    a quota problem into a hung request."""
    slept: list[float] = []
    monkeypatch.setattr("app.embedding_pipeline.embedder.time.sleep", slept.append)

    with pytest.raises(RateLimitError):
        OpenAIEmbedder(_FakeOpenAI(fail_times=99)).embed_many(_chunks(1))

    assert slept == [1.0, 2.0, 4.0]


def test_wrong_dimensionality_is_an_error_not_a_stored_vector() -> None:
    client = _FakeOpenAI()
    client.embeddings.create = lambda *, model, input: type(  # type: ignore[method-assign]
        "R", (), {
            "data": [type("I", (), {"index": 0, "embedding": [0.1] * 512})()],
            "usage": type("U", (), {"prompt_tokens": 1})(),
        },
    )()

    with pytest.raises(ValueError, match="dimensional"):
        OpenAIEmbedder(client).embed_many(_chunks(1))


def test_cost_is_priced_per_million_tokens() -> None:
    assert estimate_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimate_cost_usd(0) == 0.0


# ----------------------------------------------------------------------
# compare.py
# ----------------------------------------------------------------------
def test_cosine_similarity_matches_known_values() -> None:
    from scripts.compare import cosine_similarity

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # Scale-invariant: cosine compares direction, not magnitude.
    assert cosine_similarity([3.0, 4.0], [30.0, 40.0]) == pytest.approx(1.0)


def test_cosine_similarity_refuses_undefined_inputs() -> None:
    from scripts.compare import cosine_similarity

    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


# ----------------------------------------------------------------------
# POST /api/v1/embeddings/ingest
# ----------------------------------------------------------------------
@pytest.fixture
def ingest_client(client):
    """The real app with a fake embedder behind it.

    Overriding the dependency rather than patching the module keeps the
    route, the response model and the error boundary real; the only thing
    replaced is the part that would cost money.
    """
    from app.dependencies import get_embedder
    from app.main import app

    app.dependency_overrides[get_embedder] = lambda: OpenAIEmbedder(_FakeOpenAI())
    yield client
    app.dependency_overrides.clear()


def test_ingest_returns_a_vector_per_component_with_stats(ingest_client) -> None:
    payload = {"budgets": [json.loads(_budget().model_dump_json())]}
    response = ingest_client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 1
    assert len(body["chunks"][0]["embedding"]) == EMBEDDING_DIMENSIONS
    assert body["stats"]["total_budgets"] == 1
    assert body["stats"]["total_chunks"] == 1
    assert body["stats"]["estimated_cost_usd"] == pytest.approx(
        estimate_cost_usd(body["stats"]["total_tokens"])
    )


def test_malformed_budget_is_rejected_before_anything_is_embedded(ingest_client) -> None:
    payload = {"budgets": [{"budget_id": "B1"}]}
    assert ingest_client.post("/api/v1/embeddings/ingest", json=payload).status_code == 422


def test_provider_failure_becomes_a_500_without_leaking_the_detail(client) -> None:
    """The provider's wording can carry an organisation id or a quota
    message. It belongs in the logs, not in the response body."""
    from app.dependencies import get_embedder
    from app.main import app

    class _Exploding(_FakeOpenAI):
        pass

    exploding = _Exploding()
    exploding.embeddings.create = lambda *, model, input: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("insufficient_quota for org-secret-1234")
    )
    app.dependency_overrides[get_embedder] = lambda: OpenAIEmbedder(exploding)
    try:
        response = client.post(
            "/api/v1/embeddings/ingest",
            json={"budgets": [json.loads(_budget().model_dump_json())]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "org-secret-1234" not in response.text
