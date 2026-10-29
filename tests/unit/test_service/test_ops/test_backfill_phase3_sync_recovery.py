"""Skills phase rerun sync recovery (round-2 finding M1).

The skills phase skips a cluster whose ``SKILL.md`` already names it (the
on-disk probe), which closes the duplicate-extract race. With the sync of
written skill files placed inside the "have work" branch, a rerun after a
Ctrl-C'd skills phase found no work (every cluster skipped) and returned
before syncing, leaving the orphan ``SKILL.md`` files unindexed until a
manual ``everos cascade sync``.

The sync now runs before the scan, unconditionally, scoped to
``agent_skill``. This module pins that placement.
"""

from __future__ import annotations

from everos.service import BackfillOutcome, BackfillSummary, NullBackfillPresenter
from tests.fakes import (
    FakeEmbeddingProvider,
    InMemoryAgentSkillIndex,
    InMemoryClusterStore,
    RecordingCascade,
    backfill_deps,
    make_ops,
)


async def test_a_rerun_with_no_work_still_syncs_the_skill_files() -> None:
    """A scan that finds nothing must not skip the sync."""
    cascade = RecordingCascade()
    deps = backfill_deps(
        embedding=FakeEmbeddingProvider(),
        clusters=InMemoryClusterStore(),
        skills=InMemoryAgentSkillIndex(),
        cascade=cascade,
    )
    summary = BackfillSummary()

    outcome = await make_ops(backfill=deps).backfill(
        "skills", auto_yes=True, presenter=NullBackfillPresenter(), summary=summary
    )

    assert outcome is BackfillOutcome.COMPLETED
    assert cascade.syncs == [{"agent_skill"}]
    assert summary.skills is not None
    assert summary.skills.aborted is False
    assert summary.skills.blocked_by_capability is None
