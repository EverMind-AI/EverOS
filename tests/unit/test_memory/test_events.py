"""Domain event contracts and frozen-payload compatibility checks."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pydantic
import pytest
from everalgo.types import ChatMessage, MemCell

from everos.memory.events import (
    AgentCaseExtracted,
    AgentPipelineStarted,
    SkillClusterUpdated,
    UserPipelineStarted,
)

_FROZEN_PAYLOADS = (
    Path(__file__).resolve().parents[2]
    / "fixtures/compatibility/head_0d3b502_core_0_6_0"
)


def test_frozen_agent_case_payload_keeps_identity_and_omitted_defaults() -> None:
    payload = (_FROZEN_PAYLOADS / "agent_case_extracted.json").read_text()

    event = AgentCaseExtracted.model_validate_json(payload)

    assert event.event_id == "frozen-agent-case-event"
    assert event.ts == dt.datetime(2023, 11, 14, 22, 13, 20, tzinfo=dt.UTC)
    assert event.case_timestamp_ms == 1_700_000_000_123
    assert event.quality_score == 0.8
    assert event.task_intent == "cook risotto"
    assert (event.app_id, event.project_id) == ("default", "default")
    assert event.approach == ""
    assert event.key_insight is None
    assert event.case_gate is None


def test_frozen_skill_cluster_payload_keeps_omitted_defaults() -> None:
    payload = (_FROZEN_PAYLOADS / "skill_cluster_updated.json").read_text()

    event = SkillClusterUpdated.model_validate_json(payload)

    assert event.event_id == "frozen-skill-cluster-event"
    assert event.cluster_id == "cluster-frozen"
    assert (event.app_id, event.project_id) == ("default", "default")
    assert event.task_intent == event.approach == ""
    assert event.key_insight is None
    assert event.quality_score == 0.0
    assert event.case_timestamp_ms == 0
    assert event.case_vector is None
    assert event.case_gate is None


def test_frozen_pipeline_payload_preserves_nested_memcell() -> None:
    payload = (_FROZEN_PAYLOADS / "user_pipeline_started.json").read_text()

    event = UserPipelineStarted.model_validate_json(payload)

    assert event.event_id == "frozen-user-pipeline-event"
    assert (event.app_id, event.project_id) == ("default", "default")
    assert event.memcell_id == "mc-frozen"
    assert event.session_id == "session-frozen"
    assert event.memcell.timestamp == 1_700_000_000_123
    assert len(event.memcell.items) == 1
    message = event.memcell.items[0]
    assert isinstance(message, ChatMessage)
    assert message.id == "message-frozen"
    assert message.content == "Preserve this archived message."
    assert message.sender_name is None


def _sample_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hello",
                timestamp=1_700_000_000_000,
                sender_id="u1",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="hi back",
                timestamp=1_700_000_001_000,
                sender_id="agent",
            ),
        ],
        timestamp=1_700_000_001_000,
    )


def test_user_pipeline_started_topic_is_module_qualified() -> None:
    assert UserPipelineStarted.topic() == "everos.memory.events:UserPipelineStarted"


def test_agent_pipeline_started_topic_is_module_qualified() -> None:
    assert AgentPipelineStarted.topic() == "everos.memory.events:AgentPipelineStarted"


def test_user_pipeline_started_roundtrip_json() -> None:
    event = UserPipelineStarted(
        memcell_id="mc_a", session_id="s1", memcell=_sample_memcell()
    )
    restored = UserPipelineStarted.model_validate_json(event.model_dump_json())
    assert restored.memcell_id == "mc_a"
    assert restored.session_id == "s1"


def test_user_pipeline_started_is_frozen_and_extra_forbid() -> None:
    event = UserPipelineStarted(
        memcell_id="mc_a",
        session_id="s1",
        memcell=_sample_memcell(),
    )
    with pytest.raises(pydantic.ValidationError):
        UserPipelineStarted(  # type: ignore[call-arg]
            memcell_id="mc_a",
            session_id="s1",
            memcell=_sample_memcell(),
            extra_field=1,
        )
    with pytest.raises(pydantic.ValidationError):
        event.memcell_id = "mc_b"  # type: ignore[misc]


def test_user_pipeline_started_carries_memcell() -> None:
    event = UserPipelineStarted(
        memcell_id="mc_a",
        session_id="s1",
        memcell=_sample_memcell(),
    )
    assert event.memcell.items[0].content == "hello"
    assert event.memcell.items[1].sender_id == "agent"


def test_user_pipeline_started_nested_roundtrip_json() -> None:
    event = UserPipelineStarted(
        memcell_id="mc_a",
        session_id="s1",
        memcell=_sample_memcell(),
    )
    restored = UserPipelineStarted.model_validate_json(event.model_dump_json())
    assert restored.memcell.items[0].id == "m1"
    assert restored.memcell.items[1].content == "hi back"
    assert restored.memcell.timestamp == 1_700_000_001_000


def test_agent_case_extracted_new_fields_default() -> None:
    """approach/key_insight default so a pre-1.2.3 event payload deserializes."""
    payload = {
        "memcell_id": "m1",
        "case_entry_id": "c1",
        "task_intent": "cook risotto",
        "quality_score": 0.8,
        "case_timestamp_ms": 1_700_000_000_000,
        "agent_id": "a1",
    }
    event = AgentCaseExtracted.model_validate(payload)
    assert event.approach == ""
    assert event.key_insight is None


def test_skill_cluster_updated_new_fields_default() -> None:
    """All 6 pass-through fields default; a pre-1.2.3 payload deserializes."""
    payload = {"case_entry_id": "c1", "cluster_id": "cl1", "agent_id": "a1"}
    event = SkillClusterUpdated.model_validate(payload)
    assert event.task_intent == ""
    assert event.approach == ""
    assert event.key_insight is None
    assert event.quality_score == 0.0
    assert event.case_timestamp_ms == 0
    assert event.case_vector is None


def test_skill_cluster_updated_carries_case_vector() -> None:
    """When set, case_vector round-trips through JSON serialization."""
    payload = {
        "case_entry_id": "c1",
        "cluster_id": "cl1",
        "agent_id": "a1",
        "case_vector": [0.1, 0.2, 0.3],
    }
    event = SkillClusterUpdated.model_validate_json(
        SkillClusterUpdated.model_validate(payload).model_dump_json()
    )
    assert event.case_vector == [0.1, 0.2, 0.3]
