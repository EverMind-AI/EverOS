"""Backfill event producers carry complete case bodies, or skip missing cases."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from everos.component.utils.datetime import get_utc_now, to_timestamp_ms
from everos.infra.persistence.index import AgentCase
from everos.memory.cascade import _backfill
from everos.memory.events import SkillClusterUpdated


@pytest.mark.parametrize("approach", ["", "read + condense"])
async def test_skill_replay_snapshots_indexed_case(
    monkeypatch: pytest.MonkeyPatch, approach: str
) -> None:
    case = AgentCase(
        id="agent1_c1",
        entry_id="c1",
        owner_id="agent1",
        owner_type="agent",
        app_id="app1",
        project_id="project1",
        timestamp=get_utc_now(),
        session_id="s1",
        parent_id="mc1",
        md_path="agents/app1/project1/agent1/.cases/agent_case-2026-01-01.md",
        content_sha256="a" * 64,
        task_intent="summarise the doc",
        task_intent_tokens="summarise the doc",
        approach=approach,
        approach_tokens=approach,
        key_insight="keep the key points",
        quality_score=0.8,
        vector=[0.1] * 1024,
    )
    lookup = AsyncMock(return_value=case)
    monkeypatch.setattr(_backfill.agent_case_repo, "find_by_owner_entry", lookup)
    row = _backfill._SkillSourceRow("c1", "cl1", "agent1", "app1", "project1")

    event = await _backfill._skill_source_to_event(row)

    lookup.assert_awaited_once_with(
        "agent1", "c1", app_id="app1", project_id="project1"
    )
    assert event is not None
    restored = SkillClusterUpdated.model_validate_json(event.model_dump_json())
    assert restored.approach == approach
    assert restored.key_insight == case.key_insight
    assert restored.task_intent == case.task_intent
    assert restored.quality_score == case.quality_score
    assert restored.case_timestamp_ms == to_timestamp_ms(case.timestamp)
    assert restored.case_vector == case.vector


async def test_skill_replay_skips_stale_cluster_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _backfill.agent_case_repo, "find_by_owner_entry", AsyncMock(return_value=None)
    )
    row = _backfill._SkillSourceRow("c1", "cl1", "agent1", "app1", "project1")
    with capture_logs() as logs:
        assert await _backfill._skill_source_to_event(row) is None
    assert logs[0]["event"] == "cascade_backfill_skill_case_missing"
    assert logs[0]["log_level"] == "warning"
