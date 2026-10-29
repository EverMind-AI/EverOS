"""Round trip between everalgo ``Cluster`` and the domain cluster."""

from __future__ import annotations

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster

from everos.memory.mappers import cluster


def test_algo_cluster_round_trips_through_the_domain() -> None:
    algo = AlgoCluster(
        id="cl_1",
        centroid=np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
        count=3,
        last_ts=1_700_000_000_000,
        preview=["a"],
        members=["ep_1", "ep_2", "ep_3"],
    )

    back = cluster.to_algo(cluster.from_algo(algo))

    assert back.id == algo.id
    assert back.centroid.dtype == np.float32
    np.testing.assert_array_equal(back.centroid, algo.centroid)
    assert (back.count, back.last_ts, back.preview, back.members) == (
        algo.count,
        algo.last_ts,
        algo.preview,
        algo.members,
    )


def test_domain_centroid_is_a_plain_float_tuple() -> None:
    algo = AlgoCluster(centroid=np.zeros(2, dtype=np.float32), last_ts=0)

    centroid = cluster.from_algo(algo).centroid

    assert centroid == (0.0, 0.0)
    assert all(type(x) is float for x in centroid)
