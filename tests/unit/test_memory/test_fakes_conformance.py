"""The in-memory doubles in ``tests.fakes`` structurally satisfy their ports."""

from __future__ import annotations

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.memory.ports import (
    ClusterStore,
    EpisodeIndex,
    EpisodeStore,
    MemcellStore,
    ProfileIndex,
    ProfileStore,
)
from tests.fakes import make_deps

_DEPS = make_deps()


@pytest.mark.parametrize(
    ("port", "double"),
    [
        (EpisodeStore, _DEPS.stores.episode),
        (ProfileStore, _DEPS.stores.profile),
        (MemcellStore, _DEPS.stores.memcell),
        (ClusterStore, _DEPS.stores.cluster),
        (EpisodeIndex, _DEPS.indexes.episode),
        (ProfileIndex, _DEPS.indexes.profile),
        (EmbeddingProvider, _DEPS.embedding),
    ],
    ids=lambda v: getattr(v, "__name__", type(v).__name__),
)
def test_double_satisfies_port(port: type, double: object) -> None:
    assert isinstance(double, port)
