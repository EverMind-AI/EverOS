"""The maintenance surface the cascade worker calls must exist on every
repository it can be handed.

The worker holds the engine's own repositories (taken from the backend's
:class:`IndexRepositories`). A maintenance method missing from one of them
is unreachable in production: the first ``ensure_vector_indexes`` hook was
guarded by ``getattr(repo, ..., None)`` and silently never ran.
"""

from __future__ import annotations

import pytest

from everos.infra.persistence.index.protocols import IndexRepository
from everos.infra.persistence.lancedb import LanceRepoBase
from tests.fakes.storage import Storage

_MAINTENANCE = ("optimize", "prune", "rebuild_indexes", "ensure_vector_indexes")


@pytest.mark.parametrize("name", _MAINTENANCE)
def test_the_port_declares_the_maintenance_method(name: str) -> None:
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


def test_lance_repositories_run_the_chassis_maintenance_methods(
    storage: Storage,
) -> None:
    """Attribute presence is not enough: a ``pass`` body would satisfy the
    protocol and silently never build an index. The LanceDB repositories
    speak the contract themselves, so each maintenance method must be the
    chassis implementation, not a stub left by the contract mixin."""
    for repo in storage.repos.all():
        for name in _MAINTENANCE:
            assert getattr(type(repo), name) is getattr(LanceRepoBase, name), (
                f"{type(repo).__name__}.{name}"
            )
