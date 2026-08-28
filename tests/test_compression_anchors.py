"""Unit tests for ``AnchorDetector``: the heuristic regex library (English +
Spanish, see the module docstring for why) and the LLM-based mode's
delegation/fail-open behaviour. No FastAPI, no network.
"""

from __future__ import annotations

from app.sessions.compression.anchors import AnchorDetector, _AnchorClassification
from app.sessions.models import Message


def _user(content: str) -> Message:
    return Message(role="user", content=content)


# --- heuristic mode -----------------------------------------------------


def test_heuristic_detects_nda_in_english():
    match = AnchorDetector(mode="heuristic").detect(_user("This project is under an NDA."))
    assert match.is_anchor is True
    assert "nda" in match.matched_rules


def test_heuristic_detects_nda_in_spanish():
    match = AnchorDetector(mode="heuristic").detect(
        _user("Hemos firmado un acuerdo de confidencialidad con el cliente.")
    )
    assert match.is_anchor is True
    assert "nda" in match.matched_rules


def test_heuristic_detects_budget_locked_in_english():
    match = AnchorDetector(mode="heuristic").detect(
        _user("The budget is locked at 50000 EUR for this phase.")
    )
    assert match.is_anchor is True
    assert "budget_locked" in match.matched_rules


def test_heuristic_detects_budget_locked_in_spanish():
    match = AnchorDetector(mode="heuristic").detect(
        _user("El presupuesto está bloqueado en 50000 EUR para esta fase.")
    )
    assert match.is_anchor is True
    assert "budget_locked" in match.matched_rules


def test_heuristic_detects_compliance_regardless_of_language():
    match = AnchorDetector(mode="heuristic").detect(
        _user("Debe cumplir con GDPR y HIPAA desde el primer día.")
    )
    assert match.is_anchor is True
    assert "compliance" in match.matched_rules


def test_heuristic_ignores_plain_turns():
    match = AnchorDetector(mode="heuristic").detect(
        _user("Can we add a dashboard with basic charts?")
    )
    assert match.is_anchor is False
    assert match.matched_rules == []


def test_heuristic_dedupes_rule_when_both_language_variants_would_fire():
    # Extremely unlikely in real input, but the rule name must not repeat
    # even if both the EN and ES pattern for the same concept matched.
    match = AnchorDetector(mode="heuristic").detect(
        _user("Signed the contract. Contrato firmado.")
    )
    assert match.matched_rules.count("signed_contract") == 1


# --- llm mode -------------------------------------------------------------


class _FakeWrapper:
    def __init__(self, *, result: _AnchorClassification | None = None, raises: Exception | None = None):
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []

    def complete_structured_with_messages(self, *, messages, response_model, **kwargs):
        self.calls.append({"messages": messages, "response_model": response_model, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.result, {"model": "gpt-5-nano", "provider": "openai", "latency_ms": 1}


def test_llm_mode_uses_wrapper_and_routes_to_compression_group():
    wrapper = _FakeWrapper(result=_AnchorClassification(is_anchor=True, reason="frozen scope"))
    match = AnchorDetector(mode="llm", llm_wrapper=wrapper).detect(_user("Scope is frozen now."))

    assert match.is_anchor is True
    assert wrapper.calls[0]["model_name"] == "compression"


def test_llm_mode_returns_no_match_when_classifier_says_false():
    wrapper = _FakeWrapper(result=_AnchorClassification(is_anchor=False, reason="nothing durable"))
    match = AnchorDetector(mode="llm", llm_wrapper=wrapper).detect(_user("Can we tweak the colors?"))

    assert match.is_anchor is False


def test_llm_mode_falls_back_to_heuristic_on_wrapper_error():
    wrapper = _FakeWrapper(raises=RuntimeError("upstream LLM call failed"))
    match = AnchorDetector(mode="llm", llm_wrapper=wrapper).detect(_user("This is under an NDA."))

    # The wrapper blew up, but the heuristic path underneath still catches
    # the NDA phrase — fail open onto the cheaper strategy, not onto "never
    # an anchor".
    assert match.is_anchor is True
    assert "nda" in match.matched_rules


def test_llm_mode_falls_back_to_heuristic_when_no_wrapper_wired():
    match = AnchorDetector(mode="llm", llm_wrapper=None).detect(_user("Budget is locked at 1000 EUR."))
    assert match.is_anchor is True
