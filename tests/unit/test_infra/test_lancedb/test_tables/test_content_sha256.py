"""``content_sha256`` is a required field on every business lancedb table.

Cascade handler (16 doc §3.3) diffs by this digest to skip no-op
re-embeds. Every business schema — including ``agent_skill`` — declares
the field; daily-log kinds hash a per-handler subset of inline +
section keys, agent_skill hashes the file-level content-bearing parts.
"""

from __future__ import annotations

import datetime as dt

import pytest

from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
)

_VEC = [0.0] * 1024
_NOW = dt.datetime(2026, 5, 14, 10, 0, 0, tzinfo=dt.UTC)
_SHA = "f" * 64


def _episode() -> Episode:
    return Episode(
        id="u1_ep_1",
        entry_id="ep_20260514_0001",
        owner_id="u1",
        owner_type="user",
        session_id="s1",
        timestamp=_NOW,
        parent_type="memcell",
        parent_id="mc_1",
        sender_ids=["u1"],
        episode="hello world",
        episode_tokens="hello world",
        md_path="users/u1/episodes/episode-2026-05-14.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


def _atomic_fact() -> AtomicFact:
    return AtomicFact(
        id="u1_af_1",
        entry_id="af_20260514_0001",
        owner_id="u1",
        owner_type="user",
        session_id="s1",
        timestamp=_NOW,
        parent_type="memcell",
        parent_id="mc_1",
        sender_ids=["u1"],
        fact="x is y",
        fact_tokens="x is y",
        md_path="users/u1/.atomic_facts/atomic_fact-2026-05-14.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


def _foresight() -> Foresight:
    return Foresight(
        id="u1_fs_1",
        entry_id="fs_20260514_0001",
        owner_id="u1",
        owner_type="user",
        session_id="s1",
        timestamp=_NOW,
        parent_type="memcell",
        parent_id="mc_1",
        sender_ids=["u1"],
        foresight="user plans X",
        foresight_tokens="user plans X",
        md_path="users/u1/.foresights/foresight-2026-05-14.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


def _agent_case() -> AgentCase:
    return AgentCase(
        id="a1_ac_1",
        entry_id="ac_20260514_0001",
        owner_id="a1",
        owner_type="agent",
        session_id="s1",
        timestamp=_NOW,
        parent_type="memcell",
        parent_id="mc_1",
        quality_score=0.9,
        task_intent="scan contract",
        task_intent_tokens="scan contract",
        approach="step 1; step 2",
        approach_tokens="step 1 step 2",
        md_path="agents/a1/.cases/agent_case-2026-05-14.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


def _agent_skill() -> AgentSkill:
    return AgentSkill(
        id="a1_demo_skill",
        owner_id="a1",
        owner_type="agent",
        name="demo_skill",
        description="just a demo",
        description_tokens="just a demo",
        content="body content",
        content_tokens="body content",
        confidence=0.7,
        maturity_score=0.6,
        source_case_ids=[],
        md_path="agents/a1/agent_skills/demo_skill/SKILL.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


@pytest.mark.parametrize(
    "factory",
    [_episode, _atomic_fact, _foresight, _agent_case, _agent_skill],
    ids=["episode", "atomic_fact", "foresight", "agent_case", "agent_skill"],
)
def test_content_sha256_round_trip(factory) -> None:  # type: ignore[no-untyped-def]
    row = factory()
    assert row.content_sha256 == _SHA
    dumped = row.model_dump()
    assert dumped["content_sha256"] == _SHA
    restored = type(row).model_validate(dumped)
    assert restored.content_sha256 == _SHA


@pytest.mark.parametrize(
    "factory",
    [_episode, _atomic_fact, _foresight, _agent_case, _agent_skill],
    ids=["episode", "atomic_fact", "foresight", "agent_case", "agent_skill"],
)
def test_content_sha256_required(factory) -> None:  # type: ignore[no-untyped-def]
    """Dropping content_sha256 from the kwargs surfaces a ValidationError."""
    row = factory()
    kwargs = row.model_dump()
    del kwargs["content_sha256"]
    with pytest.raises(Exception):  # noqa: B017,PT011
        type(row).model_validate(kwargs)
