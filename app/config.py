from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # LiteLLM wrapper: tries PRIMARY_MODEL first, falls back to FALLBACK_MODEL
    # (same or different provider) on failure.
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2

    PROMPT_VERSION: str = "v1"

    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    # Semantic cache (requires Redis Stack for the RediSearch module).
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL: int = 86400
    # When True, the semantic cache LOGS potential hits but does NOT serve them.
    # Used to gather metrics before flipping the cache on in production.
    SEMANTIC_CACHE_LOG_ONLY: bool = False

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    MAX_TURNS: int = 6
    # The conversational flow uses its own prompt version, separate from the
    # single-shot PROMPT_VERSION: it needs the <project_metadata> block and a
    # different framing, versioned independently so iterating on one flow's
    # prompt never risks the other's cache/behaviour.
    CONVERSATIONAL_PROMPT_VERSION: str = "v3"
    # Hard cap per extracted attachment, in WORDS (not tokens - the number
    # means something to a non-technical user). An oversized attachment is
    # reported back as a failed attachment rather than silently truncated, so
    # the user knows content was dropped instead of getting a partial answer.
    MAX_ATTACHMENT_WORDS: int = 8000
    # The metadata extractor runs once per turn; a small/cheap model is
    # enough, and it gets its own primary/fallback pair so a flaky provider
    # doesn't break metadata refresh even though it's a "side" call.
    METADATA_EXTRACTOR_MODEL: str = "gpt-4o-mini"
    METADATA_EXTRACTOR_FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"

    # Compression (Session 5): folds evicted history into a running summary,
    # and optionally classifies anchor turns. Both are light, high-volume
    # side calls, so they get the cheapest model with structured-output
    # support (gpt-5-nano) rather than reusing PRIMARY_MODEL.
    COMPRESSION_MODEL: str = "gpt-5-nano"
    COMPRESSION_FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    # "heuristic": free regex match, no LLM call, deterministic. "llm": an
    # Instructor classifier call per evicted turn, more robust to paraphrase
    # at the cost of one extra call. Heuristic is the safe default; llm is
    # opt-in via env for sessions where recall matters more than cost.
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"

    @model_validator(mode="after")
    def validate_at_least_one_api_key(self) -> "Settings":
        """LiteLLM's Router may dispatch to either provider on fallback, so at
        least one API key must be configured or every call would fail with no
        plan B."""
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
