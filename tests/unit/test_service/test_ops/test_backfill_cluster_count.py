"""The clusters phase counts clusters of every kind across all owners."""

from __future__ import annotations

from everos.memory.models import Cluster
from everos.memory.ports import Scope
from everos.service.ops.backfill._clusters import _count_clusters
from tests.fakes import InMemoryClusterStore


def _cluster(cluster_id: str) -> Cluster:
    return Cluster(id=cluster_id, centroid=(0.0,), last_ts=0, members=["m"])


async def test_counts_user_and_agent_clusters_of_every_owner() -> None:
    store = InMemoryClusterStore()
    alice = Scope(app_id="a", project_id="p", owner_id="alice", owner_type="user")
    bot = Scope(app_id="a", project_id="p", owner_id="bot", owner_type="agent")
    store.seed(alice, "user_memory", _cluster("cl_1"), _cluster("cl_2"))
    store.seed(bot, "agent_case", _cluster("cl_3"))

    assert await _count_clusters(store) == 3


async def test_an_empty_store_counts_zero() -> None:
    assert await _count_clusters(InMemoryClusterStore()) == 0
