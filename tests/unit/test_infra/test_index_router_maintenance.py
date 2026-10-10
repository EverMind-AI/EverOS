"""The maintenance surface the cascade worker calls must exist on every
repository it can be handed.

The worker holds ``RoutedEpisodeRepository`` and friends, not the LanceDB
repository. A method added to the LanceDB layer but not forwarded here is
unreachable in production: the first ``ensure_vector_indexes`` hook was
guarded by ``getattr(repo, ..., None)`` and silently never ran.
"""

from __future__ import annotations

import pytest

from everos.core.persistence import LanceRepoBase
from everos.infra.persistence import lancedb as _lancedb
from everos.infra.persistence.index.protocols import IndexRepository
from everos.infra.persistence.index.router import RoutedIndexRepository

_MAINTENANCE = ("optimize", "prune", "rebuild_indexes", "ensure_vector_indexes")


class _Stub:
    schema = object()
    table_name = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def ensure_vector_indexes(self) -> None:
        self.calls.append("ensure_vector_indexes")


@pytest.mark.parametrize("name", _MAINTENANCE)
def test_every_repository_class_exposes_the_maintenance_method(name: str) -> None:
    for cls in (type(_lancedb.episode_repo), RoutedIndexRepository):
        assert callable(getattr(cls, name, None)), f"{cls.__name__}.{name} missing"
    assert name in IndexRepository.__protocol_attrs__  # type: ignore[attr-defined]


def test_milvus_repository_exposes_the_maintenance_methods() -> None:
    pytest.importorskip("pymilvus")
    from everos.infra.persistence.milvus import repository as milvus

    classes = [
        c
        for c in vars(milvus).values()
        if isinstance(c, type)
        and c.__module__ == milvus.__name__
        and callable(getattr(c, "rebuild_indexes", None))
    ]
    assert classes, "no Milvus repository class found"
    for cls in classes:
        for name in _MAINTENANCE:
            assert callable(getattr(cls, name, None)), f"{cls.__name__}.{name}"


async def test_router_forwards_ensure_vector_indexes_to_the_lance_repo() -> None:
    stub = _Stub()
    routed = RoutedIndexRepository(stub, milvus_repo_name="unused")  # type: ignore[arg-type]
    await routed.ensure_vector_indexes()
    assert stub.calls == ["ensure_vector_indexes"]


def test_lance_repositories_run_the_chassis_maintenance_methods() -> None:
    """Attribute presence is not enough: a ``pass`` body would satisfy the
    protocol and silently never build an index. The LanceDB singletons speak
    the contract themselves, so each maintenance method must be the chassis
    implementation, not a stub left by the contract mixin."""
    for repo in (
        _lancedb.ALL_REPOS
        if hasattr(_lancedb, "ALL_REPOS")
        else (
            _lancedb.episode_repo,
            _lancedb.atomic_fact_repo,
            _lancedb.foresight_repo,
            _lancedb.agent_case_repo,
            _lancedb.agent_skill_repo,
            _lancedb.user_profile_repo,
            _lancedb.knowledge_topic_repo,
        )
    ):
        for name in _MAINTENANCE:
            assert getattr(type(repo), name) is getattr(LanceRepoBase, name), (
                f"{type(repo).__name__}.{name}"
            )
