from __future__ import annotations

import pydantic
import pytest
from everalgo.types import ChatMessage, MemCell

from everos.memory.events import AgentPipelineStarted, UserPipelineStarted


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
