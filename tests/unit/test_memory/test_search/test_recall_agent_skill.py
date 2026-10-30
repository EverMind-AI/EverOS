"""Real-LanceDB tests for ``AgentSkillRecaller.bridged_from_cases``.

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
from everos.infra.persistence.index.adapters import AgentSkillIndexAdapter
from everos.infra.persistence.lancedb import (
    AgentSkill as LanceAgentSkill,
)
from everos.memory.ports.query import all_of, eq
from everos.memory.search.recall.agent_skill import AgentSkillRecaller
from everos.memory.search.recall.base import RecallerDeps
from tests.fakes.storage import Storage


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
    yield


def _recaller(storage: Storage) -> AgentSkillRecaller:
    return AgentSkillRecaller(
        RecallerDeps(tokenizer=_WhitespaceTokenizer()),
        AgentSkillIndexAdapter(storage.repos.agent_skill),
    )


_OWNER_WHERE = all_of(eq("owner_id", "agt"), eq("owner_type", "agent"))


async def _bridge(storage: Storage, case_scores: dict[str, float]) -> list[str]:
    got = await _recaller(storage).bridged_from_cases(
        case_scores, _OWNER_WHERE, top_k=10
    )
    return sorted(c.id for c in got.candidates)


async def test_bridge_matches_any_lineage_case(storage: Storage) -> None:
    """OR over case ids: a skill surfaces when its ``source_case_ids``
    contains at least one queried case."""
    await storage.repos.agent_skill.upsert(
        [
            _skill_row(name="s1", owner_id="agt", source_case_ids=["c_a", "c_b"]),
            _skill_row(name="s2", owner_id="agt", source_case_ids=["c_c"]),
            _skill_row(name="s3", owner_id="agt", source_case_ids=["c_d"]),
        ]
    )

    assert await _bridge(storage, {"c_a": 0.5, "c_c": 0.5}) == ["agt_s1", "agt_s2"]


async def test_bridge_respects_owner_partition(storage: Storage) -> None:
    """The ``where`` clause is AND-composed with ``array_has(...)`` — a
    skill in a different owner partition must not leak through."""
    await storage.repos.agent_skill.upsert(
        [
            _skill_row(name="s1", owner_id="agt", source_case_ids=["c_a"]),
            _skill_row(name="s1", owner_id="other", source_case_ids=["c_a"]),
        ]
    )

    assert await _bridge(storage, {"c_a": 0.5}) == ["agt_s1"]


async def test_bridge_returns_empty_for_no_cases(storage: Storage) -> None:
    """Empty input short-circuits — no index query is issued."""
    assert await _bridge(storage, {}) == []


async def test_bridge_escapes_single_quotes(storage: Storage) -> None:
    """A case id with a single quote must not break the rendered literal."""
    quoted_id = "ac_o'brien_0001"
    await storage.repos.agent_skill.upsert(
        [_skill_row(name="s1", owner_id="agt", source_case_ids=[quoted_id])]
    )

    assert await _bridge(storage, {quoted_id: 0.5}) == ["agt_s1"]


async def test_bridge_max_pools_score_across_source_cases(storage: Storage) -> None:
    """A skill learned from several matched cases takes the best case score;
    the record rides beside the candidate for the shaper."""
    await storage.repos.agent_skill.upsert(
        [
            _skill_row(name="s1", owner_id="agt", source_case_ids=["c_a", "c_b"]),
            _skill_row(name="s2", owner_id="agt", source_case_ids=["c_b", "c_z"]),
        ]
    )

    got = await _recaller(storage).bridged_from_cases(
        {"c_a": 0.9, "c_b": 0.4}, _OWNER_WHERE, top_k=10
    )

    scores = {c.id: c.score for c in got.candidates}
    assert scores == {"agt_s1": 0.9, "agt_s2": 0.4}
    assert got.stored_by_id["agt_s1"].item.source_case_ids == ["c_a", "c_b"]
