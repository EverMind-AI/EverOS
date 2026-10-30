"""everalgo ``clustering.Cluster`` <-> domain :class:`~everos.memory.models.Cluster`.

The algo cluster holds its centroid as a float32 ``numpy`` array; the domain
holds a float tuple. ``to_algo`` restores float32 so geometry merges see the
same precision the stored centroid had.
"""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster

from everos.memory.models import Cluster

_CENTROID_DTYPE = np.float32


def from_algo(algo: AlgoCluster) -> Cluster:
    return Cluster(
        id=algo.id,
        centroid=tuple(float(x) for x in np.asarray(algo.centroid).ravel()),
        count=algo.count,
        last_ts=algo.last_ts,
        preview=list(algo.preview),
        members=list(algo.members),
    )


def to_algo(cluster: Cluster) -> AlgoCluster:
    return AlgoCluster(
        id=cluster.id,
        centroid=np.asarray(cluster.centroid, dtype=_CENTROID_DTYPE),
        count=cluster.count,
        last_ts=cluster.last_ts,
        preview=list(cluster.preview),
        members=list(cluster.members),
    )
