"""Anchor detection — which turns must survive eviction.

A turn becomes an anchor when it carries a durable commitment the
conversation cannot afford to forget: signed agreements, frozen scope,
regulatory or legal context, locked budgets. The detector is intentionally
conservative — false negatives just demote a turn to the regular sliding
window (which then gets summarised), while false positives bloat the prompt
because anchors are never evicted.

Two strategies coexist (``app.config.Settings.ANCHOR_DETECTION_MODE``):

- ``"heuristic"`` (default): cheap regex over a curated phrase list. No LLM
  call, deterministic, easy to reason about.
- ``"llm"``: a binary classifier through the same Instructor primitive every
  other structured call in this service uses
  (``LLMWrapper.complete_structured_with_messages``, routed through the
  "compression" Router group, see ``app/services/llm_wrapper.py``). Robust
  to paraphrase but adds one cheap LLM call per evicted turn.

The estimator explicitly accepts descriptions in any language (the system
prompt translates the *output* to English, not the input, see
``app/guardrails/input.py``), so an English-only pattern list would be a
real gap here too: each phrase below is duplicated for Spanish.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, Field

from app.sessions.models import Message

if TYPE_CHECKING:
    from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()


# Curated phrase library. Each entry is a compiled regex matching either a
# distinctive verb phrase, a legal/compliance term, or an explicit
# commitment, in English and Spanish. The list is intentionally short — we'd
# rather miss than over-anchor.
_HEURISTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nda", re.compile(
        r"\b(nda|non[- ]?disclosure|under embargo|legal hold)\b", re.IGNORECASE)),
    ("nda", re.compile(
        r"\b(acuerdo\s+de\s+confidencialidad|bajo\s+secreto|retenci[oó]n\s+legal)\b",
        re.IGNORECASE)),
    ("signed_contract", re.compile(
        r"\b(signed|countersigned)\s+(the\s+)?(contract|sow|msa|agreement)\b", re.IGNORECASE)),
    ("signed_contract", re.compile(
        r"\b(contrato|acuerdo)\s+(firmado|refrendado)\b|\bhemos\s+firmado\s+el\s+(contrato|acuerdo)\b",
        re.IGNORECASE)),
    ("scope_frozen", re.compile(
        r"\b(scope|backlog)\s+(is\s+)?(frozen|locked|final|fixed)\b", re.IGNORECASE)),
    ("scope_frozen", re.compile(
        r"\b(alcance|backlog)\s+(est[aá]\s+)?(congelado|cerrado|bloqueado|fijado)\b",
        re.IGNORECASE)),
    ("budget_locked", re.compile(
        r"\b(budget|cap|ceiling)\s+(is\s+)?(locked|fixed|approved|capped)\s+at\b", re.IGNORECASE)),
    ("budget_locked", re.compile(
        r"\b(presupuesto|tope)\s+(est[aá]\s+)?(bloqueado|fijado|aprobado|limitado)\s+en\b",
        re.IGNORECASE)),
    ("compliance", re.compile(
        r"\b(hipaa|gdpr|rgpd|sox|pci[- ]?dss|fda|iso[- ]?27001|lopd)\b", re.IGNORECASE)),
    ("deadline_hard", re.compile(
        r"\bhard\s+deadline\b|\bmust\s+go\s+live\s+by\b", re.IGNORECASE)),
    ("deadline_hard", re.compile(
        r"\bfecha\s+l[ií]mite\s+estricta\b|\bdebe\s+estar\s+en\s+producci[oó]n\s+antes\s+de\b",
        re.IGNORECASE)),
    ("contractual", re.compile(
        r"\bcontractually\s+(bound|required|obliged)\b", re.IGNORECASE)),
    ("contractual", re.compile(
        r"\bcontractualmente\s+(obligad[oa]|vinculad[oa])\b", re.IGNORECASE)),
    ("explicit_commitment", re.compile(
        r"\b(we|the (client|customer))\s+(agreed|committed)\s+to\b", re.IGNORECASE)),
    ("explicit_commitment", re.compile(
        r"\b(acordamos|nos\s+comprometimos\s+a|el\s+cliente\s+acept[oó])\b", re.IGNORECASE)),
)


@dataclass
class AnchorMatch:
    """Result of an anchor check. Carries the rule names that fired so the
    structured log can attribute the decision."""

    is_anchor: bool
    matched_rules: list[str] = field(default_factory=list)


class _AnchorClassification(BaseModel):
    """Pydantic schema used by the LLM-based detector."""

    is_anchor: bool = Field(
        description=(
            "True if this user turn introduces a durable commitment, legal/"
            "compliance constraint, frozen scope, or locked budget that the "
            "conversation must not forget across turns."
        )
    )
    reason: str = Field(default="", max_length=200)


class AnchorDetector:
    """Decide if a turn deserves to survive eviction.

    The detector inspects the *user* turn — the assistant's reply is a
    function of it, so we only need to flag once per pair.
    """

    def __init__(
        self,
        *,
        mode: Literal["heuristic", "llm"] = "heuristic",
        llm_wrapper: "LLMWrapper | None" = None,
    ) -> None:
        self.mode = mode
        self.llm_wrapper = llm_wrapper

    def detect(self, message: Message) -> AnchorMatch:
        """Return the anchor decision plus the rules that triggered it."""
        if self.mode == "llm":
            return self._detect_llm(message)
        return self._detect_heuristic(message)

    # -- implementations -------------------------------------------------

    @staticmethod
    def _detect_heuristic(message: Message) -> AnchorMatch:
        content = message.content or ""
        matched = [name for name, pattern in _HEURISTIC_PATTERNS if pattern.search(content)]
        if matched:
            # Dedupe while preserving order — the same rule name can appear
            # twice (EN + ES variant both matching).
            matched = list(dict.fromkeys(matched))
            log.info(
                "anchor_detected", strategy="heuristic", rules=matched, content_chars=len(content)
            )
            return AnchorMatch(is_anchor=True, matched_rules=matched)
        return AnchorMatch(is_anchor=False)

    def _detect_llm(self, message: Message) -> AnchorMatch:
        if self.llm_wrapper is None:
            # Fallback: behave like heuristic if no wrapper was wired.
            return self._detect_heuristic(message)
        system_prompt = (
            "You are a classifier. Decide whether the user turn introduces a "
            "durable commitment about the project — signed agreements, frozen "
            "scope, locked budget, or legal/compliance context (NDA, HIPAA, "
            "GDPR, etc). The turn may be written in any language. Only return "
            "is_anchor=true if the turn would harm the conversation if "
            "forgotten. Otherwise return false. Reason is one short sentence."
        )
        user_message = f"User turn:\n{message.content}"
        try:
            classification, _meta = self.llm_wrapper.complete_structured_with_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=_AnchorClassification,
                model_name="compression",
                max_tokens=200,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 — fail open onto the heuristic
            log.warning(
                "anchor_llm_failed_fallback_heuristic",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return self._detect_heuristic(message)
        if classification.is_anchor:
            log.info("anchor_detected", strategy="llm", reason=classification.reason[:120])
            return AnchorMatch(is_anchor=True, matched_rules=[f"llm:{classification.reason[:60]}"])
        return AnchorMatch(is_anchor=False)
