"""Real-LanceDB tests for ``AgentSkillRecaller.fetch_by_case_ids``.

The case→skill bridge reverse-resolves skills by ``source_case_ids``
membership using DataFusion's ``array_has`` on a ``list<utf8>`` column.
These tests exercise the actual SQL ``where`` predicate (no recaller
stubs):

* OR-composition over multiple case ids,
* hits respect the partition filter (``where`` passed by the caller),
* empty case-id input short-circuits without a LanceDB call,
* case ids containing single quotes round-trip safely via the ``_q``
  escaper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.persistence.lancedb import (
    AgentSkill as LanceAgentSkill,
)
from everos.infra.persistence.lancedb import (
    agent_skill_repo,
    lancedb_manager,
)
from everos.memory.search.recall.agent_skill import AgentSkillRecaller
from everos.memory.search.recall.base import RecallerDeps


class _WhitespaceTokenizer(Tokenizer):
    """Bridge reverse-fetch never tokenises; satisfy the deps contract."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def _skill_row(
    *,
    name: str,
    owner_id: str,
    source_case_ids: list[str],
) -> LanceAgentSkill:
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
        source_case_ids=source_case_ids,
        cluster_id=None,
        md_path=f"agents/{owner_id}/skills/{name}/SKILL.md",
        content_sha256="x" * 64,
        vector=[0.0] * 1024,
    )


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate LanceDB under tmp memory root per test."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


def _recaller() -> AgentSkillRecaller:
    return AgentSkillRecaller(RecallerDeps(tokenizer=_WhitespaceTokenizer()))


_OWNER_WHERE = "owner_id = 'agt' AND owner_type = 'agent'"


async def test_fetch_by_case_ids_matches_any_lineage_case() -> None:
    """OR over case ids: a skill surfaces when its ``source_case_ids``
    contains at least one queried case."""
    await agent_skill_repo.upsert(
        [
            _skill_row(name="s1", owner_id="agt", source_case_ids=["c_a", "c_b"]),
            _skill_row(name="s2", owner_id="agt", source_case_ids=["c_c"]),
            _skill_row(name="s3", owner_id="agt", source_case_ids=["c_d"]),
        ]
    )

    got = await _recaller().fetch_by_case_ids(["c_a", "c_c"], _OWNER_WHERE, limit=10)

    assert sorted(c.id for c in got) == ["agt_s1", "agt_s2"]


async def test_fetch_by_case_ids_respects_owner_partition() -> None:
    """The ``where`` clause is AND-composed with ``array_has(...)`` — a
    skill in a different owner partition must not leak through."""
    await agent_skill_repo.upsert(
        [
            _skill_row(name="s1", owner_id="agt", source_case_ids=["c_a"]),
            _skill_row(name="s1", owner_id="other", source_case_ids=["c_a"]),
        ]
    )

    got = await _recaller().fetch_by_case_ids(["c_a"], _OWNER_WHERE, limit=10)

    assert [c.id for c in got] == ["agt_s1"]


async def test_fetch_by_case_ids_returns_empty_for_no_ids() -> None:
    """Empty input short-circuits — no LanceDB query is issued."""
    got = await _recaller().fetch_by_case_ids([], _OWNER_WHERE, limit=10)
    assert got == []


async def test_fetch_by_case_ids_escapes_single_quotes() -> None:
    """A case id with a single quote must not break the SQL literal.

    The ``_q`` escaper turns ``'`` into ``''`` (SQL standard); without it
    the where-clause would close the string literal prematurely.
    """
    quoted_id = "ac_o'brien_0001"
    await agent_skill_repo.upsert(
        [_skill_row(name="s1", owner_id="agt", source_case_ids=[quoted_id])]
    )

    got = await _recaller().fetch_by_case_ids([quoted_id], _OWNER_WHERE, limit=10)

    assert [c.id for c in got] == ["agt_s1"]


async def test_fetch_by_case_ids_carries_source_case_ids_in_metadata() -> None:
    """The full ``source_case_ids`` list must ride back in metadata so the
    manager's max-pool can score against the caller's case_score map."""
    await agent_skill_repo.upsert(
        [_skill_row(name="s1", owner_id="agt", source_case_ids=["c_a", "c_b", "c_c"])]
    )

    got = await _recaller().fetch_by_case_ids(["c_a"], _OWNER_WHERE, limit=10)

    assert len(got) == 1
    assert sorted(got[0].metadata["source_case_ids"]) == ["c_a", "c_b", "c_c"]
