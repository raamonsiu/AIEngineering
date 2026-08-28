"""Unit tests for ``resolve_tier``: the rule chain's precedence, the
English + Spanish pattern coverage, and the metadata-derived rules.
No FastAPI, no network.
"""

from __future__ import annotations

import app.sessions.tier_resolver as tier_resolver
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier, resolve_tier


def test_defaults_to_default_tier_when_nothing_matches():
    tier, rule = resolve_tier(transcript="Add a dashboard with charts.", metadata=ProjectMetadata())
    assert tier is Tier.DEFAULT
    assert rule == "default"


def test_nda_mention_in_english_resolves_to_executive():
    tier, rule = resolve_tier(
        transcript="This whole engagement is under an NDA.", metadata=ProjectMetadata()
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_nda_mention_in_spanish_resolves_to_executive():
    tier, rule = resolve_tier(
        transcript="Firmamos un acuerdo de confidencialidad con el cliente.",
        metadata=ProjectMetadata(),
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_nda_mention_in_prior_agreed_scope_also_resolves_to_executive():
    tier, rule = resolve_tier(
        transcript="Let's continue scoping the reporting module.",
        metadata=ProjectMetadata(agreed_scope="Covered under NDA signed last quarter."),
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_regulatory_mention_resolves_to_executive():
    tier, rule = resolve_tier(
        transcript="The platform must comply with HIPAA and GDPR.", metadata=ProjectMetadata()
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "regulatory_context"


def test_regulatory_mention_in_spanish_acronym_resolves_to_executive():
    tier, rule = resolve_tier(
        transcript="Debe cumplir con el RGPD desde el lanzamiento.", metadata=ProjectMetadata()
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "regulatory_context"


def test_regulatory_mention_in_mentioned_technologies_resolves_to_executive():
    tier, rule = resolve_tier(
        transcript="Continue the previous scope.",
        metadata=ProjectMetadata(mentioned_technologies=["HIPAA-compliant storage"]),
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "regulatory_context"


def test_single_technical_keyword_is_not_enough_for_developer_tier():
    tier, rule = resolve_tier(
        transcript="We already run everything on docker.", metadata=ProjectMetadata()
    )
    assert tier is Tier.DEFAULT
    assert rule == "default"


def test_two_distinct_technical_keywords_resolve_to_developer():
    tier, rule = resolve_tier(
        transcript="We run docker and kubernetes with a kafka event bus.",
        metadata=ProjectMetadata(),
    )
    assert tier is Tier.DEVELOPER
    assert rule == "technical_audience"


def test_spanish_microservices_keyword_counts_towards_developer_tier():
    tier, rule = resolve_tier(
        transcript="Usamos microservicios sobre kubernetes para todo el backend.",
        metadata=ProjectMetadata(),
    )
    assert tier is Tier.DEVELOPER
    assert rule == "technical_audience"


def test_small_team_resolves_to_pm():
    tier, rule = resolve_tier(
        transcript="Just a small internal tool.",
        metadata=ProjectMetadata(assumed_team_size=2),
    )
    assert tier is Tier.PM
    assert rule == "low_budget_pm"


def test_team_size_above_threshold_does_not_resolve_to_pm():
    tier, rule = resolve_tier(
        transcript="Just a small internal tool.",
        metadata=ProjectMetadata(assumed_team_size=3),
    )
    assert tier is Tier.DEFAULT
    assert rule == "default"


def test_nda_takes_precedence_over_a_small_team():
    tier, rule = resolve_tier(
        transcript="This is under an NDA.",
        metadata=ProjectMetadata(assumed_team_size=1),
    )
    assert tier is Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_a_failing_predicate_is_skipped_in_favour_of_the_next_rule(monkeypatch):
    def _boom(_ctx):
        raise RuntimeError("boom")

    # ``_RULES`` binds each predicate at module load time, so patching the
    # standalone ``_has_nda`` function after the fact wouldn't reach the
    # bound reference — replace the rule tuple entry instead.
    broken_rule = tier_resolver.TierRule("nda_detected", Tier.EXECUTIVE, _boom)
    monkeypatch.setattr(tier_resolver, "_RULES", (broken_rule, *tier_resolver._RULES[1:]))

    tier, rule = resolve_tier(
        transcript="This is under an NDA and also relies on HIPAA data handling.",
        metadata=ProjectMetadata(),
    )

    # nda_detected's predicate is broken, but regulatory_context still fires.
    assert tier is Tier.EXECUTIVE
    assert rule == "regulatory_context"
