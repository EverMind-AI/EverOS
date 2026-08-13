"""Round-2 finding #6: Phase-3 idempotency across Ctrl-C.

``extract_agent_skill`` writes ``SKILL.md`` first and only reaches
LanceDB via ``_sync_new_skill_files`` afterwards. If Phase 3 is
interrupted between the md write and the sync (Ctrl-C during
``engine.wait_idle`` or between ``engine.stop`` and the sync call),
markdown-only skills are left behind. Round-1's ``_scan_skill_source``
gated cluster replay only on ``agent_skill_repo.count_in_cluster > 0``
— on the next Phase-3 run those md-only clusters looked "unextracted"
and got re-emitted, causing duplicate skill extraction.

Round-2 adds a disk check via :func:`_skill_md_exists_for_cluster`:
if any ``SKILL.md`` under the agent's ``skills/`` directory carries
this ``cluster_id`` in its frontmatter, the cluster is skipped even
when LanceDB has no row yet.
"""

from __future__ import annotations

from pathlib import Path

from everos.memory.cascade import _backfill


class _StubAgentSkillRepo:
    """Return count_in_cluster == 0 for every cluster — LanceDB never
    saw the skill get indexed. The disk check must still skip the
    cluster."""

    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int:
        return 0


class _StubClusterRepo:
    """Yield one agent-owned cluster whose members Phase 3 would
    otherwise re-emit."""

    def __init__(self, cluster_id: str, members: list[str]) -> None:
        self._cluster_id = cluster_id
        self._members = members

    async def list_distinct_owners(self):  # type: ignore[no-untyped-def]
        return [("agent_a", "agent", "default", "default")]

    async def list_for_owner(
        self,
        owner_id: str,
        kind: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ):  # type: ignore[no-untyped-def]
        import numpy as np
        from everalgo.clustering import Cluster as AlgoCluster

        return [
            AlgoCluster(
                id=self._cluster_id,
                centroid=np.zeros(1, dtype=np.float32),
                count=len(self._members),
                last_ts=0,
                preview=[],
                members=list(self._members),
            )
        ]


def _write_skill_md(
    root: Path, agent_id: str, skill_name: str, cluster_id: str
) -> Path:
    """Write a minimal SKILL.md whose frontmatter carries ``cluster_id``."""
    from everos.core.persistence.memory_root import app_dir_name, project_dir_name
    from everos.infra.persistence.markdown import AgentSkillFrontmatter

    skill_dir = (
        root
        / app_dir_name("default")
        / project_dir_name("default")
        / "agents"
        / agent_id
        / AgentSkillFrontmatter.SKILLS_CONTAINER_NAME
        / f"{AgentSkillFrontmatter.SKILL_DIR_PREFIX}{skill_name}"
    )
    skill_dir.mkdir(parents=True, exist_ok=True)
    md_path = skill_dir / AgentSkillFrontmatter.SKILL_MAIN_FILENAME
    frontmatter = (
        "---\n"
        f"type: agent_skill\n"
        f"name: {skill_name}\n"
        f"description: test skill\n"
        f"confidence: 0.9\n"
        f"maturity_score: 0.5\n"
        f"cluster_id: {cluster_id}\n"
        f"---\n"
        "body\n"
    )
    md_path.write_text(frontmatter, encoding="utf-8")
    return md_path


async def test_scan_skill_source_skips_when_md_exists_but_lancedb_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """SKILL.md on disk for cluster ``c_stuck`` → scan skips it even
    though LanceDB reports zero skills for that cluster.

    Without the disk check the same cluster would re-emit into
    Phase 3 and produce a duplicate skill on the next drain. Round-2
    closes that window by consulting the filesystem before deciding
    the cluster needs a replay event.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    _write_skill_md(
        tmp_path, agent_id="agent_a", skill_name="doubled", cluster_id="c_stuck"
    )

    monkeypatch.setattr(
        _backfill,
        "agent_skill_repo",
        _StubAgentSkillRepo(),
    )
    monkeypatch.setattr(
        _backfill,
        "cluster_repo",
        _StubClusterRepo(cluster_id="c_stuck", members=["case_1", "case_2"]),
    )

    rows = await _backfill._scan_skill_source()

    # Empty output despite the LanceDB probe returning 0 is the
    # invariant: it means the disk check ran and matched. Without
    # round-2's fix the same setup would emit one row per cluster
    # member, driving duplicate skill extraction on the next drain.
    assert rows == []


async def test_scan_skill_source_emits_when_no_md_or_lancedb_row(
    tmp_path: Path, monkeypatch
) -> None:
    """Neither disk nor LanceDB has the cluster's skill → Phase 3
    correctly emits one row per member so replay can extract it."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    monkeypatch.setattr(
        _backfill,
        "agent_skill_repo",
        _StubAgentSkillRepo(),
    )
    monkeypatch.setattr(
        _backfill,
        "cluster_repo",
        _StubClusterRepo(cluster_id="c_fresh", members=["case_1", "case_2"]),
    )

    rows = await _backfill._scan_skill_source()

    assert len(rows) == 2
    assert {r.case_entry_id for r in rows} == {"case_1", "case_2"}
    assert {r.cluster_id for r in rows} == {"c_fresh"}


async def test_skill_md_exists_for_cluster_matches_frontmatter_cluster_id(
    tmp_path: Path, monkeypatch
) -> None:
    """Direct helper contract: a SKILL.md whose frontmatter carries
    the queried cluster_id is a hit; a mismatch is a miss."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.core.persistence import MemoryRoot

    _write_skill_md(tmp_path, agent_id="agent_a", skill_name="alpha", cluster_id="c_A")
    _write_skill_md(tmp_path, agent_id="agent_a", skill_name="beta", cluster_id="c_B")

    root = MemoryRoot.resolve()

    assert await _backfill._skill_md_exists_for_cluster(
        cluster_id="c_A",
        agent_id="agent_a",
        app_id="default",
        project_id="default",
        memory_root=root,
    )
    assert await _backfill._skill_md_exists_for_cluster(
        cluster_id="c_B",
        agent_id="agent_a",
        app_id="default",
        project_id="default",
        memory_root=root,
    )
    assert not await _backfill._skill_md_exists_for_cluster(
        cluster_id="c_ghost",
        agent_id="agent_a",
        app_id="default",
        project_id="default",
        memory_root=root,
    )
    # Missing skills dir for a fresh agent → False, no exception.
    assert not await _backfill._skill_md_exists_for_cluster(
        cluster_id="c_A",
        agent_id="agent_b",
        app_id="default",
        project_id="default",
        memory_root=root,
    )
