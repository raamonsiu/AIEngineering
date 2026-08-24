"""In-memory session store.

A plain ``dict[str, Session]`` indexed by ``session_id``. Not thread-safe:
fine under a single ``uvicorn`` worker; with multiple workers each worker
would have its own copy of the store, which breaks the conversational
guarantee. Documented here so the limitation is visible at the
abstraction's surface rather than discovered in production.
"""

from __future__ import annotations

from app.sessions.models import DEFAULT_MAX_TURNS, ConversationHistory, Session


class SessionStore:
    """Process-local session registry, indexed by ``session_id``."""

    def __init__(self, *, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._sessions: dict[str, Session] = {}
        self.max_turns = max_turns

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self.max_turns))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def __len__(self) -> int:
        return len(self._sessions)
