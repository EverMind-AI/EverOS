"""Verify OME strategy registration is unconditional (capability check is body-guard).

``_get_engine()`` builds the singleton :class:`OfflineEngine` and registers
every OME strategy exactly once. Four strategies re-embed or depend on a
cluster produced by re-embedding (``trigger_profile_clustering``,
``trigger_skill_clustering``, ``extract_agent_skill``, ``reflect_episodes``);
these are now registered regardless of embed availability. Each guards
its own body via :func:`get_embedding_capability` at execution time so a
runtime tier upgrade (Tier 1 → Tier 2) picks up on the next dispatch
without an engine restart — OME's registry is frozen after ``start()``,
which is why a registration-time gate cannot self-heal.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability

_svc = importlib.import_module("everos.service.memorize")

_ALWAYS = {
    "extract_atomic_facts",
    "extract_foresight",
    "extract_agent_case",
    "extract_user_profile",
}
_REQUIRE_EMBED = {
    "trigger_profile_clustering",
    "trigger_skill_clustering",
    "extract_agent_skill",
    "reflect_episodes",
}


class _StubEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        return [0.0]


@pytest.fixture(autouse=True)
def _reset_engine_singleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a fresh ``OfflineEngine`` per test, rooted at a scratch dir."""
    from everos.core.persistence import MemoryRoot

    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(_svc, "_ome_engine", None, raising=False)


def _set_embed_available(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    import everos.component.embedding.accessor as embed_accessor

    monkeypatch.setattr(
        embed_accessor,
        "_capability",
        EmbeddingCapability(provider=_StubEmbeddingProvider() if available else None),
    )


def _registered_names() -> set[str]:
    engine = _svc._get_engine()
    return {meta.name for meta in engine._registry.all()}


def test_all_strategies_registered_when_embed_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed available → all 8 strategies register."""
    _set_embed_available(monkeypatch, available=True)
    names = _registered_names()
    assert names >= _ALWAYS | _REQUIRE_EMBED


def test_all_strategies_registered_when_embed_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed unavailable → registration is still unconditional.

    The four embed-requiring strategies stay in the registry so that a
    runtime tier upgrade (edit everos.toml + reload settings) takes
    effect on the next dispatch without a server restart. Each strategy
    body guards on :func:`get_embedding_capability` and no-ops when
    unavailable — see the per-strategy tests for that half of the
    contract.
    """
    _set_embed_available(monkeypatch, available=False)
    names = _registered_names()
    assert names >= _ALWAYS | _REQUIRE_EMBED
