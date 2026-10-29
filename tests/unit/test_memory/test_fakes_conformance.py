"""The in-memory doubles in ``tests.fakes`` structurally satisfy their ports."""

from __future__ import annotations

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.memory.ports import (
    AgentCaseIndex,
    AgentCaseStore,
    AgentSkillIndex,
    AgentSkillStore,
    AtomicFactIndex,
    AtomicFactStore,
    ClusterStore,
    EpisodeIndex,
    EpisodeStore,
    ForesightStore,
    MemcellStore,
    ProfileIndex,
    ProfileStore,
    ReflectionReportStore,
)
from tests.fakes import make_deps

_DEPS = make_deps()


@pytest.mark.parametrize(
    ("port", "double"),
    [
        (EpisodeStore, _DEPS.stores.episode),
        (AtomicFactStore, _DEPS.stores.atomic_fact),
        (ForesightStore, _DEPS.stores.foresight),
        (ProfileStore, _DEPS.stores.profile),
        (AgentCaseStore, _DEPS.stores.agent_case),
        (AgentSkillStore, _DEPS.stores.agent_skill),
        (MemcellStore, _DEPS.stores.memcell),
        (ClusterStore, _DEPS.stores.cluster),
        (ReflectionReportStore, _DEPS.stores.reflection_report),
        (EpisodeIndex, _DEPS.indexes.episode),
        (AtomicFactIndex, _DEPS.indexes.atomic_fact),
        (AgentCaseIndex, _DEPS.indexes.agent_case),
        (AgentSkillIndex, _DEPS.indexes.agent_skill),
        (ProfileIndex, _DEPS.indexes.profile),
        (EmbeddingProvider, _DEPS.embedding),
    ],
    ids=lambda v: getattr(v, "__name__", type(v).__name__),
)
def test_double_satisfies_port(port: type, double: object) -> None:
    assert isinstance(double, port)
