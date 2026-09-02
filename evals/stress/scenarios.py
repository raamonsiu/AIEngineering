"""Three synthetic multi-turn conversations, each built to stress one way a
CAG system can lose information.

Why synthetic and scripted, rather than replayed real conversations: the
question is "at which turn does the system forget X", and that only has an
answer if we know exactly when X was said and in what words. A real
transcript gives you realism and no ground truth; this gives up realism to
get a fact tracker, which is the thing the measurement needs.

The three profiles target three different failure modes:

- ``growing``       — monotonically accumulating requirements. Nothing is
                      ever retracted, so every fact declared at turn k is
                      still true at turn 20. Any loss is pure forgetting.
- ``pivot``         — at turn 5 the stack is replaced outright. Tests
                      whether memory *updates* or merely *accumulates*.
- ``contradiction`` — the same quantity is asserted twice with different
                      values (turn 3 and turn 8), both phrased to trip the
                      anchor detector. Tests which one wins, and whether
                      the system notices there is a conflict at all.

Each profile declares facts with an ``expectation``:

- ``persist``    — this must still be findable in the session's memory later.
                   Not finding it is forgetting.
- ``superseded`` — this *was* true and has since been replaced. Still
                   finding it is a different defect: stale memory. The
                   metric scores presence either way (see
                   ``evals.stress.metrics.MemoryDriftMetric``); the sign of
                   that score is read off ``expectation``, which keeps the
                   metric free of scenario-specific judgement.

``aliases`` exist because the matcher is deliberately a case-insensitive
substring match (determinism over sophistication — no embeddings, no
LLM-as-judge). Without aliases the metric would measure *formatting*: a
summarizer that writes "30,000 EUR" where the transcript said "30000 EUR"
has not forgotten the budget, but an exact match would score it as drift.
Listing the surface forms a faithful system might legitimately use keeps
the metric measuring memory instead of punctuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Expectation = Literal["persist", "superseded"]

# The three memory slots a fact can legitimately live in. The default for a
# fact is "any of them": the CAG's whole design is that a fact may migrate
# from the sliding window into the summary, or be rescued into an anchor,
# and insisting on a particular slot would score a *working* system as
# broken.
DEFAULT_SLOTS: tuple[str, ...] = ("summary", "anchors", "metadata")


@dataclass(frozen=True)
class Fact:
    """One tracked claim: what was said, when, and whether it should last."""

    key: str
    value: str
    introduced_at_turn: int
    expectation: Expectation = "persist"
    aliases: tuple[str, ...] = ()
    # Turn at which a ``superseded`` fact stopped being true. None for
    # ``persist`` facts.
    superseded_at_turn: int | None = None
    where: tuple[str, ...] = DEFAULT_SLOTS

    @property
    def needles(self) -> tuple[str, ...]:
        """Every surface form that counts as "this fact is present"."""
        return (self.value, *self.aliases)


@dataclass(frozen=True)
class Turn:
    index: int  # 1-based
    transcript: str


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    turns: tuple[Turn, ...]
    facts: tuple[Fact, ...] = field(default_factory=tuple)

    def facts_live_at(self, turn_index: int) -> tuple[Fact, ...]:
        """Facts already introduced by the time ``turn_index`` completed.

        A fact introduced *on* the turn being measured counts: it was in
        this turn's own transcript, so a system that cannot find it one
        instant later has not merely forgotten, it never recorded it.
        """
        return tuple(f for f in self.facts if f.introduced_at_turn <= turn_index)

    @property
    def opening_transcript(self) -> str:
        """Turn 1's text, reused as the carrier for the attachment sweep so
        that sweep varies attachment size and nothing else."""
        return self.turns[0].transcript


def _turns(*transcripts: str) -> tuple[Turn, ...]:
    return tuple(Turn(index=i, transcript=t) for i, t in enumerate(transcripts, start=1))


# ----------------------------------------------------------------------
# Profile 1: a project that only ever grows
# ----------------------------------------------------------------------
GROWING = Scenario(
    key="growing",
    description=(
        "Nimbus, a fleet-maintenance SaaS. Each turn adds a coherent new "
        "requirement and retracts nothing, so every fact stays true to the "
        "end. Measures the cost curve and whether the project's own name "
        "survives twenty turns."
    ),
    turns=_turns(
        "We are scoping a new platform called Nimbus: a web SaaS for fleet maintenance "
        "scheduling, replacing the spreadsheets our client uses today. Give us a first estimate.",
        "Add single sign-on. The client's IT department requires SAML 2.0 against their "
        "existing Azure AD directory.",
        "The platform has to be multi-tenant: one deployment serving around forty "
        "independent customer workspaces, with data isolation between them.",
        "We also need a full audit log: every change to a maintenance record must be "
        "traceable to a user and a timestamp, and kept for seven years.",
        "Operations wants CSV export of any report grid, plus a scheduled weekly email "
        "delivering the same data to a distribution list.",
        "Billing will go through Stripe, with per-seat subscriptions and annual invoicing "
        "for the larger accounts.",
        "The mobile side is a read-only React Native app for mechanics working in the workshop.",
        "That mobile app needs an offline mode: the mechanics lose signal in the "
        "underground service bay and still have to read job cards.",
        "Add a parts inventory module, with low-stock alerts and per-supplier reorder "
        "thresholds.",
        "Integrate with the client's existing SAP instance so purchase orders raised in "
        "Nimbus appear in their finance system.",
        "The dashboard needs real-time vehicle telemetry ingestion, roughly two thousand "
        "events per minute at peak.",
        "Add role-based access control with four roles: admin, fleet manager, mechanic and "
        "read-only auditor.",
        "We want a public REST API for customers who prefer their own front-end, "
        "rate-limited per tenant.",
        "Localisation for launch: Spanish, English and German, with more languages added later.",
        "Accessibility has to reach WCAG 2.1 AA. The client supplies the public sector and "
        "it is a procurement requirement.",
        "Add predictive maintenance suggestions derived from the historical failure data "
        "they already hold.",
        "The client wants a white-label option: custom logo, colour scheme and domain per "
        "tenant workspace.",
        "We need a data migration tool for the twelve years of spreadsheets they currently keep.",
        "Add a customer-facing status page showing scheduled maintenance windows and "
        "current system health.",
        "Final pass: please re-estimate the whole platform with everything we have "
        "discussed across this conversation.",
    ),
    facts=(
        Fact(key="project_name", value="Nimbus", introduced_at_turn=1),
        Fact(key="sso", value="SAML", introduced_at_turn=2, aliases=("SAML 2.0", "single sign-on")),
        Fact(key="tenancy", value="multi-tenant", introduced_at_turn=3, aliases=("multitenant", "multi tenant")),
        Fact(key="audit", value="audit log", introduced_at_turn=4, aliases=("audit trail", "auditing")),
        Fact(key="export", value="CSV", introduced_at_turn=5, aliases=("CSV export",)),
        Fact(key="billing", value="Stripe", introduced_at_turn=6),
        Fact(key="erp", value="SAP", introduced_at_turn=10),
        Fact(key="accessibility", value="WCAG", introduced_at_turn=15, aliases=("WCAG 2.1", "accessibility")),
    ),
)


# ----------------------------------------------------------------------
# Profile 2: a project that pivots at turn 5
# ----------------------------------------------------------------------
PIVOT = Scenario(
    key="pivot",
    description=(
        "Halcyon, a booking portal that starts as a React web app and is "
        "replaced wholesale by a Flutter mobile app at turn 5. Measures "
        "whether metadata updates cleanly or whether mentioned_technologies "
        "simply accumulates both stacks."
    ),
    turns=_turns(
        "We are planning Halcyon, a customer-facing appointment booking portal. The "
        "front-end will be built in React, with a Node backend and PostgreSQL.",
        "Add a calendar view with drag-and-drop rescheduling of existing appointments.",
        "Bookings need to support recurring appointments and a configurable cancellation policy.",
        "We need payment capture at booking time, with partial refunds when a customer cancels.",
        "Change of direction from the client: drop the React web front-end entirely. Halcyon "
        "will be a Flutter mobile app for iOS and Android instead. The web portal is cancelled, "
        "do not estimate it.",
        "The Flutter app needs push notifications for booking reminders, twenty-four hours ahead.",
        "Offline-first behaviour: the app must queue bookings made without connectivity and "
        "sync them when the device is back online.",
        "Add biometric login on the mobile app, Face ID and fingerprint.",
        "The booking flow has to work for three appointment types with different durations and "
        "staff requirements.",
        "Staff get their own view in the same Flutter app to manage their daily schedule.",
        "Add in-app chat between a customer and the assigned staff member for a booking.",
        "We need deep links so a reminder notification opens the specific booking directly.",
        "Analytics: track booking funnel drop-off per appointment type.",
        "The app must support Spanish and English, switching with the device locale.",
        "Add a loyalty scheme: every tenth booking is discounted automatically.",
        "Integrate with Google Calendar and Apple Calendar so customers can mirror bookings.",
        "We need an admin back-office, and this one genuinely is a web application.",
        "Add waitlist handling: when a slot frees up, notify waitlisted customers in order.",
        "The client wants App Store and Play Store release management included in the estimate.",
        "Final pass: re-estimate Halcyon end to end with the stack we actually settled on.",
    ),
    facts=(
        Fact(key="project_name", value="Halcyon", introduced_at_turn=1),
        Fact(
            key="stack_superseded",
            value="React",
            introduced_at_turn=1,
            expectation="superseded",
            superseded_at_turn=5,
        ),
        Fact(key="stack_current", value="Flutter", introduced_at_turn=5),
    ),
)


# ----------------------------------------------------------------------
# Profile 3: a project that contradicts itself
# ----------------------------------------------------------------------
#
# Both budget turns are phrased to match the anchor detector's
# "budget_locked" pattern ("budget is locked at ..."). That is deliberate:
# if only the second one anchored, the test would merely show recency
# winning. Making both anchor-eligible asks the sharper question — when two
# anchors contradict each other, does anything resolve the conflict, or do
# both sit in the prompt at once?
CONTRADICTION = Scenario(
    key="contradiction",
    description=(
        "Perseus, an HR onboarding tool whose budget is locked at 30000 EUR "
        "on turn 3 and re-locked at 80000 EUR on turn 8. Both turns are "
        "anchor-eligible. Measures which figure survives, which is promoted "
        "to an anchor, and which ends up in the summary."
    ),
    turns=_turns(
        "We are scoping Perseus, an internal HR onboarding workflow tool for a company of "
        "about six hundred employees.",
        "It should handle document collection, e-signature and equipment provisioning for "
        "each new hire.",
        "The budget is locked at 30000 EUR for the whole project. Please keep the estimate "
        "within that ceiling.",
        "Add an onboarding checklist that each new hire's manager can track to completion.",
        "We need integration with the existing payroll system to create the employee record.",
        "Add automated email sequences: welcome message, first-week agenda, thirty-day check-in.",
        "The tool should generate the employment contract from a template and send it for "
        "e-signature.",
        "Correction from the client's finance team: the budget is locked at 80000 EUR, not "
        "the figure we gave you earlier. Please re-estimate against the new ceiling.",
        "Add a reporting view showing average time-to-productive per department.",
        "Equipment provisioning must raise a ticket in their existing Jira Service Management.",
        "We need a self-service portal where the new hire uploads their own documents.",
        "Add support for contractors as well as employees, with a reduced checklist.",
        "The contract templates have to exist in Spanish and English.",
        "Add a manager dashboard showing every in-flight onboarding at a glance.",
        "We need GDPR-compliant retention: candidate documents deleted after the legal period.",
        "Add bulk onboarding for the graduate intake, forty people starting the same week.",
        "Integrate with their single sign-on so new hires use one account from day one.",
        "Add an offboarding flow reusing the same checklist engine in reverse.",
        "The client wants an audit report of every document signed, exportable as PDF.",
        "Final pass: re-estimate Perseus in full, and state clearly which budget ceiling you "
        "are working to.",
    ),
    facts=(
        Fact(key="project_name", value="Perseus", introduced_at_turn=1),
        Fact(
            key="budget_superseded",
            value="30000",
            introduced_at_turn=3,
            expectation="superseded",
            superseded_at_turn=8,
            aliases=("30,000", "30.000", "30k", "30 000"),
        ),
        Fact(
            key="budget_current",
            value="80000",
            introduced_at_turn=8,
            aliases=("80,000", "80.000", "80k", "80 000"),
        ),
    ),
)


SCENARIOS: dict[str, Scenario] = {s.key: s for s in (GROWING, PIVOT, CONTRADICTION)}


def get_scenarios(keys: list[str]) -> list[Scenario]:
    unknown = sorted(set(keys) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenario(s): {unknown}; known: {sorted(SCENARIOS)}")
    return [SCENARIOS[k] for k in keys]
