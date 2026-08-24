"""In-memory conversational state for the multi-turn estimation flow.

Two structures per session, deliberately kept separate rather than merged
into one blob:

- ``ConversationHistory``, the raw ``(user, assistant)`` turn pairs that get
  replayed to the LLM, bounded by a sliding window of ``max_turns``. Turns
  that fall out of the window are gone from history.
- ``ProjectMetadata`` (``app/schemas/estimation.py``), durable facts about
  the project, updated after every turn and never truncated. This is what
  makes the sliding window safe to use as the *only* history strategy: a fact
  established on turn 1 still informs turn 20 even after turn 1 itself has
  scrolled out of the window, because it lives on ``Session.project_metadata``
  instead of inside the window.

Storage is a plain process-local dict, no database, no Redis. Accepted
volatility: a session is short-lived working memory for one active
conversation, not a system of record, and this phase of the project is about
the CAG architecture (separating history from memory), not about surviving a
restart. Persistence/federation across processes belongs to the deployment
module.
"""

from __future__ import annotations

from uuid import uuid4

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectMetadata, ProjectType

DEFAULT_MAX_TURNS = 6


class ConversationHistory:
    """Sliding window over the last ``max_turns`` (user, assistant) pairs.

    The system prompt is deliberately NOT stored here: ``EstimationService``
    rebuilds it fresh from the session's current ``ProjectMetadata`` on every
    call and passes it into ``to_messages_list``. That makes "always preserve
    the system prompt" automatic rather than a truncation special-case to get
    right.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        self._turns.append((user_content, assistant_content))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def to_messages_list(self, system_prompt: str) -> list[dict]:
        """The full ``messages`` array ready for the LLM call: system prompt
        first, then the surviving turns in order."""
        messages = [{"role": "system", "content": system_prompt}]
        for user_content, assistant_content in self._turns:
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def __len__(self) -> int:
        return len(self._turns)


class Session:
    """One conversation's worth of state: history + metadata + the typed
    selectors the form-based endpoint normally requires up front.

    The typed selectors (``project_type``, ``detail_level``, ``output_format``)
    are optional on each turn of the multi-turn endpoint but remembered here:
    set once (or defaulted on creation) and only overwritten when a turn
    explicitly supplies a new value. This keeps the "typed request, not free
    chat" contract from the single-shot endpoint instead of dropping it for
    the conversational flow.
    """

    def __init__(self, *, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self.session_id = str(uuid4())
        self.history = ConversationHistory(max_turns=max_turns)
        self.project_metadata = ProjectMetadata()
        self.project_type = ProjectType.WEB_SAAS
        self.detail_level = DetailLevel.MEDIUM
        self.output_format = OutputFormat.PHASES_TABLE


class SessionStore:
    """Process-local session registry, indexed by ``session_id``."""

    def __init__(self, *, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._sessions: dict[str, Session] = {}
        self.max_turns = max_turns

    def create(self) -> Session:
        session = Session(max_turns=self.max_turns)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)
