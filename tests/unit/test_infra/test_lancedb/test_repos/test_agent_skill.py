"""Tests for :class:`everos.infra.persistence.lancedb._AgentSkillRepo`.

Real LanceDB under ``tmp_path`` (no mocks) — these tests exercise the
SQL ``where`` predicate, cosine ``distance_type`` ranking, and
``_distance`` stripping that the repo owns. Strategy-level routing
across these methods is covered separately in
``tests/unit/test_memory/test_strategies/test_extract_agent_skill.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence.lancedb import (
    AgentSkill as LanceAgentSkill,
)
from everos.infra.persistence.lancedb import (
    agent_skill_repo,
    lancedb_manager,
)


def _skill_row(
    *,
    name: str,
    owner_id: str,
    cluster_id: str,
    vector: list[float],
) -> LanceAgentSkill:
    """Minimal AgentSkill row sufficient to land in LanceDB for repo tests."""
    return LanceAgentSkill(
        id=f"{owner_id}_{name}",
        owner_id=owner_id,
        owner_type="agent",
        name=name,
        description=f"desc {name}",
        description_tokens=f"desc {name}",
        content=f"body of {name}",
        content_tokens=f"body of {name}",
        confidence=0.7,
        maturity_score=0.6,
        source_case_ids=[],
        cluster_id=cluster_id,
        md_path=f"agents/{owner_id}/skills/{name}/SKILL.md",
        content_sha256="x" * 64,
        vector=vector,
    )


@pytest.fixture
async def _real_lancedb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a clean LanceDB rooted under ``tmp_path`` for one test."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def test_count_in_cluster_isolates_owner_and_cluster(
    _real_lancedb: None,
) -> None:
    """``count_in_cluster`` returns only rows matching both filters."""
    await agent_skill_repo.upsert(
        [
            _skill_row(name="s1", owner_id="a", cluster_id="cl_x", vector=[0.1] * 1024),
            _skill_row(name="s2", owner_id="a", cluster_id="cl_x", vector=[0.2] * 1024),
            _skill_row(
                name="other_cluster",
                owner_id="a",
                cluster_id="cl_y",
                vector=[0.3] * 1024,
            ),
            _skill_row(
                name="other_owner",
                owner_id="b",
                cluster_id="cl_x",
                vector=[0.4] * 1024,
            ),
        ]
    )

    assert (
        await agent_skill_repo.count_in_cluster(owner_id="a", cluster_id="cl_x")
    ) == 2


async def test_find_in_cluster_returns_typed_rows_no_ranking(
    _real_lancedb: None,
) -> None:
    """Scalar fetch within one cluster; capped at ``limit`` regardless of order."""
    await agent_skill_repo.upsert(
        [
            _skill_row(name="s1", owner_id="a", cluster_id="cl_x", vector=[0.1] * 1024),
            _skill_row(name="s2", owner_id="a", cluster_id="cl_x", vector=[0.2] * 1024),
            _skill_row(name="s3", owner_id="a", cluster_id="cl_x", vector=[0.3] * 1024),
            _skill_row(
                name="other_cluster",
                owner_id="a",
                cluster_id="cl_y",
                vector=[0.4] * 1024,
            ),
        ]
    )

    got = await agent_skill_repo.find_in_cluster(
        owner_id="a", cluster_id="cl_x", limit=2
    )
    assert len(got) == 2
    assert {s.name for s in got}.issubset({"s1", "s2", "s3"})
    assert all(s.owner_id == "a" and s.cluster_id == "cl_x" for s in got)


async def test_find_topk_relevant_in_cluster_ranks_by_cosine(
    _real_lancedb: None,
) -> None:
    """LanceDB native ``nearest_to + distance_type('cosine')`` ordering."""
    near = [1.0] + [0.0] * 1023
    far = [0.0] * 1023 + [1.0]
    medium = [0.7, 0.7] + [0.0] * 1022
    await agent_skill_repo.upsert(
        [
            _skill_row(name="near", owner_id="a", cluster_id="cl_x", vector=near),
            _skill_row(name="far", owner_id="a", cluster_id="cl_x", vector=far),
            _skill_row(name="medium", owner_id="a", cluster_id="cl_x", vector=medium),
            # Different cluster — must not leak.
            _skill_row(name="other", owner_id="a", cluster_id="cl_y", vector=near),
            # Different owner — must not leak either.
            _skill_row(name="near", owner_id="b", cluster_id="cl_x", vector=near),
        ]
    )

    got = await agent_skill_repo.find_topk_relevant_in_cluster(
        owner_id="a", cluster_id="cl_x", query_vector=near, top_k=2
    )
    assert [s.name for s in got] == ["near", "medium"]


async def test_find_topk_relevant_in_cluster_raises_on_empty_vector(
    _real_lancedb: None,
) -> None:
    """Empty ``query_vector`` is a caller-side error — the repo refuses."""
    await agent_skill_repo.upsert(
        [
            _skill_row(name="s1", owner_id="a", cluster_id="cl_x", vector=[0.1] * 1024),
        ]
    )
    with pytest.raises(ValueError, match="query_vector must be non-empty"):
        await agent_skill_repo.find_topk_relevant_in_cluster(
            owner_id="a", cluster_id="cl_x", query_vector=[], top_k=2
        )
