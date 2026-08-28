"""Unit tests for ``CompressionPolicy``: the sliding-window trim, anchor
promotion, and the running summary — now that trimming lives here rather
than in ``ConversationHistory.append`` (see ``app/sessions/models.py``).
No FastAPI, no network.
"""

from __future__ import annotations

from app.sessions.compression.anchors import AnchorDetector
from app.sessions.compression.policy import CompressionPolicy, apply_compression
from app.sessions.models import ConversationHistory
from tests._session_test_helpers import FakeLLMWrapper


class _FakeSummarizer:
    """Records every call instead of hitting the real prompt/LLM plumbing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def summarize(self, *, previous_summary, evicted):
        self.calls.append({"previous_summary": previous_summary, "evicted": evicted})
        evicted_ids = ",".join(m.content[:12] for m in evicted)
        return f"{previous_summary or ''}|{evicted_ids}".strip("|")


def _policy(detector: AnchorDetector | None = None, summarizer=None) -> tuple[CompressionPolicy, object]:
    summarizer = summarizer or _FakeSummarizer()
    detector = detector or AnchorDetector(mode="heuristic")
    return CompressionPolicy(anchor_detector=detector, summarizer=summarizer), summarizer


def test_should_compress_is_false_under_the_cap():
    history = ConversationHistory(max_turns=2)
    history.append(user="a", assistant="b")
    policy, _ = _policy()
    assert policy.should_compress(history) is False


def test_apply_is_a_noop_under_the_cap():
    history = ConversationHistory(max_turns=2)
    history.append(user="a", assistant="b")
    policy, summarizer = _policy()

    policy.apply(history)

    assert len(history.messages) == 2
    assert summarizer.calls == []


def test_apply_evicts_the_oldest_pair_into_the_summary_when_not_an_anchor():
    history = ConversationHistory(max_turns=1)
    history.append(user="turn zero", assistant="answer zero")
    history.append(user="turn one", assistant="answer one")
    policy, summarizer = _policy()

    policy.apply(history)

    assert len(history.messages) == 2
    assert history.messages[0].content == "turn one"
    assert history.anchors == []
    assert len(summarizer.calls) == 1
    assert [m.content for m in summarizer.calls[0]["evicted"]] == ["turn zero", "answer zero"]
    assert history.summary is not None


def test_apply_rescues_an_anchor_instead_of_summarizing_it():
    history = ConversationHistory(max_turns=1)
    history.append(user="Budget is locked at 20000 EUR.", assistant="Noted, budget locked.")
    history.append(user="turn one", assistant="answer one")
    policy, summarizer = _policy()

    policy.apply(history)

    assert len(history.messages) == 2
    assert len(history.anchors) == 2
    assert history.anchors[0].content == "Budget is locked at 20000 EUR."
    assert summarizer.calls == []
    assert history.summary is None


def test_apply_integrates_previous_summary_across_multiple_passes():
    history = ConversationHistory(max_turns=1)
    policy, summarizer = _policy()

    history.append(user="turn zero", assistant="answer zero")
    history.append(user="turn one", assistant="answer one")
    policy.apply(history)
    first_summary = history.summary

    history.append(user="turn two", assistant="answer two")
    policy.apply(history)

    assert len(summarizer.calls) == 2
    assert summarizer.calls[1]["previous_summary"] == first_summary
    assert history.summary != first_summary


def test_apply_logs_each_pair_independently_when_mixing_anchors_and_summary():
    history = ConversationHistory(max_turns=1)
    history.append(user="turn zero", assistant="answer zero")
    history.append(user="Scope is frozen now.", assistant="Noted, scope frozen.")
    history.append(user="turn two", assistant="answer two")
    policy, summarizer = _policy()

    policy.apply(history)

    assert len(history.messages) == 2
    assert history.messages[0].content == "turn two"
    assert len(history.anchors) == 2
    assert history.anchors[0].content == "Scope is frozen now."
    assert len(summarizer.calls) == 1
    assert [m.content for m in summarizer.calls[0]["evicted"]] == ["turn zero", "answer zero"]


def test_apply_compression_wrapper_wires_heuristic_detector_and_summarizer_by_default():
    history = ConversationHistory(max_turns=1)
    history.append(user="turn zero", assistant="answer zero")
    history.append(user="turn one", assistant="answer one")
    wrapper = FakeLLMWrapper()

    apply_compression(history, llm_wrapper=wrapper)

    assert len(history.messages) == 2
    assert history.summary is not None
    compression_calls = [c for c in wrapper.calls if c["model_name"] == "compression"]
    assert len(compression_calls) == 1
