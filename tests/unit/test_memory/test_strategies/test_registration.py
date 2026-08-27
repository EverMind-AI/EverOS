"""Test strategy package exports and OME engine registration."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from everos.memory.strategies import (
    extract_agent_case,
    extract_agent_skill,
    extract_atomic_facts,
    extract_decision,
    extract_foresight,
    extract_principles,
    extract_user_profile,
    reflect_decisions,
    reflect_episodes,
    trigger_decision_clustering,
    trigger_profile_clustering,
    trigger_skill_clustering,
)


def test_strategies_are_re_exported_from_package() -> None:
    for fn, name in [
        (extract_atomic_facts, "extract_atomic_facts"),
        (extract_decision, "extract_decision"),
        (extract_foresight, "extract_foresight"),
        (extract_agent_case, "extract_agent_case"),
        (trigger_skill_clustering, "trigger_skill_clustering"),
        (extract_agent_skill, "extract_agent_skill"),
        (trigger_profile_clustering, "trigger_profile_clustering"),
        (trigger_decision_clustering, "trigger_decision_clustering"),
        (extract_user_profile, "extract_user_profile"),
        (extract_principles, "extract_principles"),
        (reflect_decisions, "reflect_decisions"),
        (reflect_episodes, "reflect_episodes"),
    ]:
        assert fn.meta.name == name


async def test_get_engine_registers_all_strategies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.core.persistence import MemoryRoot

    svc = importlib.import_module("everos.service.memorize")

    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)

    engine = svc._get_engine()
    names = {m.name for m in engine._registry.all()}
    assert names == {
        "extract_atomic_facts",
        "extract_decision",
        "extract_foresight",
        "extract_agent_case",
        "trigger_skill_clustering",
        "extract_agent_skill",
        "trigger_profile_clustering",
        "trigger_decision_clustering",
        "extract_user_profile",
        "extract_principles",
        "reflect_decisions",
        "reflect_episodes",
    }
