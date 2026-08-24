"""Unit tests for the session domain models, no FastAPI, no LLM, no network.

These exercise the two invariants the multi-turn flow depends on:
``ConversationHistory``'s window trimming, and ``ProjectMetadata.merge_with``'s
deterministic merge (the safety net for an LLM extractor forgetting to
restate a fact).
"""

from __future__ import annotations

from app.sessions.models import ConversationHistory, ProjectMetadata, Session


def test_conversation_history_trims_to_max_turns():
    history = ConversationHistory(max_turns=2)
    for i in range(4):
        history.append(user=f"user {i}", assistant=f"assistant {i}")

    assert len(history) == 2
    assert len(history.messages) == 4
    # The oldest pairs are dropped from the front: turns 0 and 1 are gone.
    assert history.messages[0].content == "user 2"
    assert history.messages[-1].content == "assistant 3"


def test_conversation_history_preserves_role_alternation_after_trim():
    history = ConversationHistory(max_turns=1)
    history.append(user="first user", assistant="first assistant")
    history.append(user="second user", assistant="second assistant")

    roles = [m.role for m in history.messages]
    assert roles == ["user", "assistant"]


def test_to_messages_excludes_the_system_prompt():
    history = ConversationHistory(max_turns=6)
    history.append(user="hello", assistant="world")

    messages = history.to_messages()

    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    assert all(m["role"] != "system" for m in messages)


def test_project_metadata_is_empty_by_default():
    assert ProjectMetadata().is_empty() is True


def test_project_metadata_is_not_empty_once_a_field_is_set():
    assert ProjectMetadata(project_name="Aurora").is_empty() is False


def test_merge_with_preserves_prior_scalars_when_update_is_null():
    """The core bug this merge exists to prevent: a turn where the extractor
    doesn't restate a fact must not erase it."""
    previous = ProjectMetadata(project_name="Aurora", assumed_team_size=3)
    update = ProjectMetadata()

    merged = previous.merge_with(update)

    assert merged.project_name == "Aurora"
    assert merged.assumed_team_size == 3


def test_merge_with_overwrites_scalar_when_update_provides_new_value():
    previous = ProjectMetadata(assumed_team_size=3)
    update = ProjectMetadata(assumed_team_size=5)

    merged = previous.merge_with(update)

    assert merged.assumed_team_size == 5


def test_merge_with_unions_technologies_case_insensitively_without_duplicates():
    previous = ProjectMetadata(mentioned_technologies=["React"])
    update = ProjectMetadata(mentioned_technologies=["react", "PostgreSQL"])

    merged = previous.merge_with(update)

    assert merged.mentioned_technologies == ["React", "PostgreSQL"]


def test_merge_with_unions_constraints_and_rejected_options():
    previous = ProjectMetadata(explicit_constraints=["GDPR compliance"], rejected_options=["Microservices"])
    update = ProjectMetadata(explicit_constraints=["GDPR compliance"], rejected_options=["Serverless"])

    merged = previous.merge_with(update)

    assert merged.explicit_constraints == ["GDPR compliance"]
    assert merged.rejected_options == ["Microservices", "Serverless"]


def test_session_defaults_to_web_saas_medium_phases_table():
    session = Session()

    assert session.project_type.value == "web_saas"
    assert session.detail_level.value == "medium"
    assert session.output_format.value == "phases_table"
    assert session.project_metadata.is_empty()
    assert len(session.history) == 0
