"""Hybrid memory compression for long conversations (Session 5).

Three pieces work together:

- ``AnchorDetector`` decides whether a turn carries durable information the
  conversation should never forget (signed contract, frozen scope, agreed
  budget, regulatory mention). Anchors are excluded from eviction.
- ``CumulativeSummarizer`` folds older non-anchor turns into a running
  free-text summary, kept on ``ConversationHistory.summary``.
- ``CompressionPolicy`` is the orchestrator: it inspects the history after
  each ``append`` and decides what (if anything) to compress, including the
  sliding-window trim itself (see ``app.sessions.models`` for why that's no
  longer ``ConversationHistory.append``'s job).

``ConversationHistory.to_messages()`` composes the output as:

    [summary_envelope?] + anchors_in_order + recent_sliding_window
"""

from app.sessions.compression.anchors import AnchorDetector, AnchorMatch
from app.sessions.compression.policy import CompressionPolicy, apply_compression
from app.sessions.compression.summarizer import CumulativeSummarizer

__all__ = [
    "AnchorDetector",
    "AnchorMatch",
    "CompressionPolicy",
    "CumulativeSummarizer",
    "apply_compression",
]
