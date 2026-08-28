"""Derive the audience tier at runtime from the conversation context.

Tiers shape the prompt's framing (executive bullets vs PM milestones vs
developer-grade detail vs default). The resolver is a precedence-ordered
chain of pure-function rules over the current turn's transcript plus the
session's accumulated ``ProjectMetadata``. Each rule returns
``True``/``False``; the first match decides the tier. If none matches, the
tier is ``DEFAULT``. The resolver returns both the tier and the **rule
name** that fired, that explainability is what lets the response report
"executive (nda_detected)" instead of just "executive".

The estimator explicitly accepts descriptions in any language (the system
prompt translates the *output* to English, not the input, see
``app/guardrails/input.py``), so an English-only pattern list would be a
real gap here too: the NDA/regulatory patterns are duplicated for Spanish,
the other language this service is actually used in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import structlog

from app.sessions.models import ProjectMetadata

log = structlog.get_logger()


class Tier(str, Enum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata


@dataclass
class TierRule:
    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


# --- Predicates --------------------------------------------------------------

_NDA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(nda|non[- ]?disclosure|confidential|under embargo|legal hold)\b", re.IGNORECASE),
    re.compile(
        r"\b(acuerdo\s+de\s+confidencialidad|confidencial|bajo\s+secreto|"
        r"retenci[oó]n\s+legal)\b",
        re.IGNORECASE,
    ),
)
_REGULATORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(hipaa|gdpr|rgpd|sox|pci[- ]?dss|fda|iso[- ]?27001|ccpa|lopd)\b", re.IGNORECASE
    ),
)
_DEV_KEYWORDS_PATTERN = re.compile(
    r"\b(docker|kubernetes|k8s|microservice|microservices|microservicio|microservicios|"
    r"terraform|iac|helm|grpc|graphql|kafka|airflow|spark|rabbitmq)\b",
    re.IGNORECASE,
)


def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _has_nda(ctx: ResolutionContext) -> bool:
    if _matches_any(_NDA_PATTERNS, ctx.transcript):
        return True
    if ctx.metadata.agreed_scope and _matches_any(_NDA_PATTERNS, ctx.metadata.agreed_scope):
        return True
    return False


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    if _matches_any(_REGULATORY_PATTERNS, ctx.transcript):
        return True
    if ctx.metadata.agreed_scope and _matches_any(_REGULATORY_PATTERNS, ctx.metadata.agreed_scope):
        return True
    for tech in ctx.metadata.mentioned_technologies:
        if _matches_any(_REGULATORY_PATTERNS, tech):
            return True
    return False


def _is_small_team(ctx: ResolutionContext) -> bool:
    return ctx.metadata.assumed_team_size is not None and ctx.metadata.assumed_team_size <= 2


def _technical_audience(ctx: ResolutionContext) -> bool:
    hits = _DEV_KEYWORDS_PATTERN.findall(ctx.transcript)
    # Require at least two distinct technical keywords to flip — a single
    # mention ("we run on docker") is too noisy for tier promotion.
    return len({hit.lower() for hit in hits}) >= 2


# --- Rule chain --------------------------------------------------------------

_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(*, transcript: str, metadata: ProjectMetadata) -> tuple[Tier, str]:
    """Return ``(tier, rule_name)`` by evaluating the rule chain in order,
    first hit wins. Falls back to ``(Tier.DEFAULT, "default")`` when no rule
    matches. A rule whose predicate raises is logged and skipped rather than
    aborting resolution for the remaining rules."""
    ctx = ResolutionContext(transcript=transcript, metadata=metadata)

    for rule in _RULES:
        try:
            if rule.predicate(ctx):
                log.info("tier_resolved", tier=rule.tier.value, rule=rule.name)
                return rule.tier, rule.name
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "tier_rule_predicate_failed",
                rule=rule.name,
                error_type=type(exc).__name__,
                error=str(exc)[:120],
            )

    log.info("tier_resolved", tier=Tier.DEFAULT.value, rule="default")
    return Tier.DEFAULT, "default"
