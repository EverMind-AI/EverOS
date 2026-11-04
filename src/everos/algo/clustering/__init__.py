"""Online incremental clustering operators — pure-function state-in / state-out.

Public API:

* :func:`cluster_by_geometry` — cosine similarity + time-window filter + threshold; no LLM.
* :func:`acluster_by_llm` — embedding top-K recall + fast-path skip + LLM ranking; native async.
* :class:`Cluster` — frozen value object the caller threads through and persists as list[Cluster].

Caller responsibilities: embedding (any np.ndarray dimension is accepted as long as it is
consistent across calls), persistence (Cluster.model_dump() / model_validate()), and
read-modify-write serialisation if multiple writers share a list.
"""

from __future__ import annotations

import logging

from everos.algo.clustering.algorithm import acluster_by_llm, cluster_by_geometry
from everos.algo.clustering.state import Cluster

__all__ = [
    "Cluster",
    "acluster_by_llm",
    "cluster_by_geometry",
]

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())
