"""FastAPI dependency factories for shared singletons."""

from __future__ import annotations

from functools import lru_cache

import redis
import structlog
from openai import OpenAI

from app.cache.semantic import EstimationSemanticCache
from app.config import get_settings
from app.ingest.anonymization.mapping_store import JsonMappingStore
from app.ingest.anonymization.pseudonymizer import ConsistentPseudonymizer
from app.ingest.catalog import DataCatalog, load_catalog
from app.ingest.orchestrator import IngestionPipeline
from app.services.cache import EstimationCache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper
from app.sessions import SessionStore
from pathlib import Path

log = structlog.get_logger()


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        metadata_extractor_model=settings.METADATA_EXTRACTOR_MODEL,
        metadata_extractor_fallback_model=settings.METADATA_EXTRACTOR_FALLBACK_MODEL,
        compression_model=settings.COMPRESSION_MODEL,
        compression_fallback_model=settings.COMPRESSION_FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
    )


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Lazy OpenAI client used by ``check_input`` (Moderation API) and the
    semantic cache (Embeddings API)."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Build the semantic cache, swallowing setup errors so the rest of the
    pipeline keeps working if Redis Stack / RediSearch is not available
    (e.g. running on vanilla redis:7-alpine)."""
    settings = get_settings()
    if get_openai_client() is None:
        log.warning("semantic_cache_disabled", reason="no_openai_key")
        return None

    try:
        from redisvl.utils.vectorize import OpenAITextVectorizer

        vectorizer = OpenAITextVectorizer(
            model=settings.EMBEDDING_MODEL,
            api_config={"api_key": settings.OPENAI_API_KEY},
        )
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
            log_only=settings.SEMANTIC_CACHE_LOG_ONLY,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "semantic_cache_disabled",
            reason="setup_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None


@lru_cache
def get_session_store() -> SessionStore:
    """Process-local, in-memory session registry (see ``app/sessions/models.py``
    for why no database/Redis is involved). ``lru_cache`` makes this a
    singleton for the process's lifetime, same as the other dependencies
    here."""
    return SessionStore(max_turns=get_settings().MAX_TURNS)


@lru_cache
def get_estimation_service() -> EstimationService:
    settings = get_settings()
    return EstimationService(
        llm_wrapper=get_llm_wrapper(),
        exact_cache=get_cache(),
        semantic_cache=get_semantic_cache(),
        openai_client=get_openai_client(),
        prompt_version=settings.PROMPT_VERSION,
        conversational_prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION,
        max_attachment_words=settings.MAX_ATTACHMENT_WORDS,
        anchor_detection_mode=settings.ANCHOR_DETECTION_MODE,
    )


# ----------------------------------------------------------------------
# RAG ingest subsystem (Session 6)
# ----------------------------------------------------------------------


@lru_cache
def get_data_catalog() -> DataCatalog:
    """Load the data catalog once per process.

    Cached because it is read on every indexing run and never changes
    without a deploy — it is a versioned artefact in the repository, not
    runtime state.
    """
    return load_catalog(Path(get_settings().DATA_CATALOG_PATH))


@lru_cache
def get_mapping_store() -> JsonMappingStore:
    return JsonMappingStore(Path(get_settings().PSEUDONYM_MAPPING_PATH))


@lru_cache
def get_pseudonymizer() -> ConsistentPseudonymizer:
    """Build the pseudonymiser once: it owns a spaCy model whose load
    costs seconds, and the offline pipeline calls it per document."""
    settings = get_settings()
    return ConsistentPseudonymizer(
        get_mapping_store(),
        locale=settings.PSEUDONYM_LOCALE,
        score_threshold=settings.PII_SCORE_THRESHOLD,
    )


def get_ingestion_pipeline() -> IngestionPipeline:
    """NOT cached, unlike the other factories.

    An indexing run mutates nothing on the pipeline object, but the
    catalog it is built from is the thing an operator edits between runs.
    A cached pipeline would keep serving a stale catalog until the process
    restarted, which is exactly the silent-staleness failure the catalog
    exists to prevent.
    """
    settings = get_settings()
    return IngestionPipeline(
        get_data_catalog(),
        corpus_root=Path(settings.CORPUS_ROOT),
        pseudonymizer=get_pseudonymizer(),
    )
