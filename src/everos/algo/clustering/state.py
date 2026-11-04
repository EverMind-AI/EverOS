"""Cluster value object — frozen Pydantic model; caller persists list[Cluster] across calls.

Each Cluster carries its own centroid/count/last_ts/preview. The clustering operators treat an
incoming item as a size-1 Cluster (BIRCH CF-tree N=1 equivalence), so input and existing
elements share one type.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


class Cluster(BaseModel):
    """Frozen value object representing one cluster snapshot.

    ``members`` carries caller-supplied entity ids; the algorithm appends them on merge but never
    inspects their semantics.

    For an incoming item, build a size-1 Cluster (count=1 default; preview=[item_text] when
    acluster_by_llm will be called).
    """

    id: str | None = (
        None  # caller-supplied business id; algo only passes it through, never mints
    )
    centroid: np.ndarray
    count: int = 1
    last_ts: int  # ms epoch
    preview: list[str] = []
    members: list[
        str
    ] = []  # caller-supplied entity ids (e.g. MemCell.id); algo appends on merge, no cap

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
