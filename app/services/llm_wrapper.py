"""LiteLLM-backed wrapper that adds provider fallback, cost tracking and
structured logging to every LLM call in the estimator.

Design notes
------------
- The wrapper exposes two primitives:
  - ``complete()``: free-text answer, cached by the exact-match cache.
  - ``complete_structured()``: returns a validated Pydantic model via Instructor,
    re-prompting on validator errors up to ``max_retries`` times. Caching for
    this path lives in ``app/services/estimation.py`` (the service owns the
    pipeline order: guardrails -> caches -> LLM -> output guardrail -> cache write).
- The Router is configured with two deployments under the same ``model_name``
  ("estimator") so LiteLLM can switch from primary to fallback transparently:
    1. Try the primary model.
    2. On a transient error, retry it up to ``num_retries`` times.
    3. If it still fails, fall over to the fallback deployment.
- Instructor wraps the *Router's* ``completion`` (not bare ``litellm.completion``)
  so structured calls get the same primary->fallback escalation as free-text
  ones. Wrapping ``litellm.completion`` directly would silently drop the
  fallback guarantee for the only endpoint this service exposes.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, TypeVar

import instructor
import structlog
from litellm import Router
from pydantic import BaseModel

from app.constants import MODELS_PRICING
from app.prompts.loader import render_metadata_extraction_prompt
from app.schemas.estimation import ProjectMetadata
from app.services.cache import EstimationCache

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


def _normalise_model_name(model: str) -> str:
    """Strip provider prefixes like ``anthropic/`` that LiteLLM may emit."""
    return model.split("/", 1)[1] if "/" in model else model


def _provider_from_model(model: str) -> str:
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3")):
        return "openai"
    return "unknown"


def _lookup_pricing(model: str) -> dict[str, float]:
    """Match a (possibly versioned, e.g. gpt-4o-mini-2024-07-18) model name
    against the known pricing table by longest matching prefix."""
    name = _normalise_model_name(model)
    if name in MODELS_PRICING:
        return MODELS_PRICING[name]
    matches = [key for key in MODELS_PRICING if name.startswith(key)]
    if matches:
        return MODELS_PRICING[max(matches, key=len)]
    return {"input": 0.0, "output": 0.0}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _lookup_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def _extract_delta(chunk: Any) -> str:
    """Pull the text delta out of a LiteLLM streaming chunk."""
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError):
        return ""
    return getattr(delta, "content", None) or ""


class LLMWrapper:
    """Unified LLM client with cache, provider fallback, and cost tracking."""

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        primary_model: str,
        fallback_model: str,
        timeout: int,
        num_retries: int,
        cache: EstimationCache,
    ):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.num_retries = num_retries
        self.cache = cache

        self.router = Router(
            model_list=[
                {
                    "model_name": "estimator",
                    "litellm_params": {
                        "model": primary_model,
                        "api_key": openai_api_key,
                        "timeout": timeout,
                    },
                },
                {
                    "model_name": "estimator",
                    "litellm_params": {
                        "model": fallback_model,
                        "api_key": anthropic_api_key,
                        "timeout": timeout,
                    },
                },
            ],
            fallbacks=[{"estimator": ["estimator"]}],
            num_retries=num_retries,
        )

        self._instructor = instructor.from_litellm(self.router.completion)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Single LLM call returning a free-text answer, with exact-match cache."""
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=self.primary_model,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(cache_key)
        if cached:
            return {**cached, "cache_hit": True}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        log.info("llm_call_started", mode="blocking", model=self.primary_model)
        t0 = time.perf_counter()
        try:
            response = self.router.completion(
                model="estimator", messages=messages, max_tokens=max_tokens
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        result = self._normalise_response(response, latency_ms=latency_ms)
        log.info(
            "llm_call_completed",
            model=result["model"],
            provider=result["provider"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            cost_usd=result["cost_usd"],
            latency_ms=latency_ms,
            finish_reason=result["finish_reason"],
        )
        self.cache.set(cache_key, result)
        return {**result, "cache_hit": False}

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        max_tokens: int = 4000,
        max_retries: int = 6,
    ) -> tuple[T, dict[str, Any]]:
        """Single system+user turn. Thin wrapper around
        ``complete_structured_with_messages`` for the single-shot endpoint."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.complete_structured_with_messages(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def complete_structured_with_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_tokens: int = 4000,
        max_retries: int = 6,
    ) -> tuple[T, dict[str, Any]]:
        """Run the LLM with Instructor over an arbitrary ``messages`` array and
        return ``(model_instance, meta)``.

        Takes the full array (rather than a single system+user pair) so a
        multi-turn caller can pass prior conversation turns ahead of the
        current one. ``meta`` carries ``model``, ``provider``, ``cost_usd`` and
        ``latency_ms`` so the caller can report what the call actually cost.
        Instructor re-prompts the LLM up to ``max_retries`` times when a
        Pydantic validator raises, feeding the ``ValueError`` message back to
        the model.
        """
        log.info(
            "llm_structured_call_started",
            model=self.primary_model,
            response_model=response_model.__name__,
        )
        t0 = time.perf_counter()
        try:
            result, completion = self._instructor.chat.completions.create_with_completion(
                model="estimator",
                messages=messages,
                response_model=response_model,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_structured_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        model = _normalise_model_name(getattr(completion, "model", self.primary_model))
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        meta = {
            "model": model,
            "provider": _provider_from_model(model),
            "cost_usd": _estimate_cost(model, input_tokens, output_tokens),
            "latency_ms": latency_ms,
        }
        log.info("llm_structured_call_completed", **meta)
        return result, meta

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        meta_holder: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive. A cache hit replays the cached text
        as a single chunk so the client UX stays consistent either way.

        Only the exact-match cache applies here. The semantic cache is keyed by
        a typed ``EstimationRequest`` (its bucket includes project type, detail
        level and output format), which this free-text path doesn't have.

        If ``meta_holder`` is passed it is populated in place with ``cache_hit``,
        ``cost_usd``, ``model`` and ``provider`` once the generator is
        exhausted : callers that need that metadata (e.g. the SSE router, to
        emit a final event) read it after the ``for`` loop ends.
        """
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=self.primary_model,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(cache_key)
        if cached:
            log.info("stream_cache_hit", chars=len(cached.get("estimation", "")))
            if meta_holder is not None:
                meta_holder.update(
                    cache_hit=True,
                    cost_usd=cached.get("cost_usd", 0.0),
                    model=cached.get("model", self.primary_model),
                    provider=cached.get("provider", _provider_from_model(self.primary_model)),
                )
            yield cached.get("estimation", "")
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        log.info("llm_stream_started", model=self.primary_model)
        t0 = time.perf_counter()
        full_text: list[str] = []
        usage_chunk: Any = None
        model_seen: str | None = None
        try:
            response = self.router.completion(
                model="estimator",
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in response:
                delta = _extract_delta(chunk)
                if delta:
                    full_text.append(delta)
                    yield delta
                # The final chunk carries cumulative usage (both providers do
                # this via LiteLLM's stream_options).
                if getattr(chunk, "usage", None):
                    usage_chunk = chunk.usage
                if getattr(chunk, "model", None):
                    model_seen = chunk.model
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_stream_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        rendered = "".join(full_text)
        model = _normalise_model_name(model_seen or self.primary_model)
        provider = _provider_from_model(model)
        input_tokens = getattr(usage_chunk, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage_chunk, "completion_tokens", 0) or 0
        cost_usd = _estimate_cost(model, input_tokens, output_tokens)
        log.info(
            "llm_stream_completed",
            latency_ms=latency_ms,
            chars=len(rendered),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

        self.cache.set(
            cache_key,
            {
                "estimation": rendered,
                "model": model,
                "provider": provider,
                "finish_reason": "stop",
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
        )

        if meta_holder is not None:
            meta_holder.update(
                cache_hit=False, cost_usd=cost_usd, model=model, provider=provider
            )

    def extract_metadata(
        self,
        *,
        current: ProjectMetadata,
        user_turn: str,
        assistant_summary: str,
        max_retries: int = 2,
    ) -> ProjectMetadata:
        """Update ``ProjectMetadata`` after a session turn via a small,
        dedicated LLM call (the "LLM extractor" approach, chosen over a regex
        heuristic because this conversation is free-form and bilingual).

        Fails open: if the call or validation fails after retries, the
        *previous* metadata is returned unchanged rather than raising, losing
        a metadata update for one turn is much cheaper than 502ing an
        otherwise-successful estimation over a side call.
        """
        prompt = render_metadata_extraction_prompt(
            current_metadata=current, user_turn=user_turn, assistant_summary=assistant_summary
        )
        try:
            result, _completion = self._instructor.chat.completions.create_with_completion(
                model="estimator",
                messages=[{"role": "user", "content": prompt}],
                response_model=ProjectMetadata,
                max_tokens=500,
                max_retries=max_retries,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - fail open, see docstring
            log.warning(
                "metadata_extraction_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return current

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_response(response: Any, *, latency_ms: int) -> dict[str, Any]:
        choice = response.choices[0]
        finish_reason = (choice.finish_reason or "stop").lower()
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        model = _normalise_model_name(response.model)
        return {
            "estimation": choice.message.content or "",
            "model": model,
            "provider": _provider_from_model(model),
            "finish_reason": finish_reason,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "latency_ms": latency_ms,
            "cost_usd": _estimate_cost(model, input_tokens, output_tokens),
        }
