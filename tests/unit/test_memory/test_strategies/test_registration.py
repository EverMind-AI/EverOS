"""Test strategy package exports and OME engine registration."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from everos.memory.strategies import (
    extract_agent_case,
    extract_agent_skill,
    extract_atomic_facts,
    extract_foresight,
    extract_user_profile,
    trigger_profile_clustering,
    trigger_skill_clustering,
)


def test_strategies_are_re_exported_from_package() -> None:
    for fn, name in [
        (extract_atomic_facts, "extract_atomic_facts"),
        (extract_foresight, "extract_foresight"),
        (extract_agent_case, "extract_agent_case"),
        (trigger_skill_clustering, "trigger_skill_clustering"),
        (extract_agent_skill, "extract_agent_skill"),
        (trigger_profile_clustering, "trigger_profile_clustering"),
        (extract_user_profile, "extract_user_profile"),
    ]:
        assert fn._ome_strategy_meta.name == name  # type: ignore[attr-defined]


async def test_get_engine_registers_all_strategies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.core.persistence import MemoryRoot

    svc = importlib.import_module("everos.service.memorize")

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)

    engine = svc._get_engine()
    names = {m.name for m in engine._registry.all()}  # noqa: SLF001 — test introspection
    assert names == {
        "extract_atomic_facts",
        "extract_foresight",
        "extract_agent_case",
        "trigger_skill_clustering",
        "extract_agent_skill",
        "trigger_profile_clustering",
        "extract_user_profile",
    }
