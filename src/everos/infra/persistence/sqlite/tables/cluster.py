"""``cluster`` — persisted snapshot of one ``everalgo.clustering.Cluster``.

Mirrors the algo-side frozen value object (centroid + count + last_ts +
preview) plus everos engineering metadata (``owner_id`` / ``owner_type``
/ ``kind``) so a single SQLite table can hold both the user-memory cluster
track (episode embeddings) and the agent-case cluster track (task_intent
embeddings). The ``members`` field on the algo type is persisted in the
sibling :class:`ClusterMember` table to keep the relation queryable from
both directions (forward by ``cluster_id``, reverse by ``(member_type,
member_id)``).
"""

from __future__ import annotations

from sqlalchemy import Index, LargeBinary

from everos.component.utils.datetime import UtcDatetime
from everos.core.persistence.sqlite import BaseTable, Field
from everos.core.persistence.sqlite.base import UtcDateTimeColumn


class Cluster(BaseTable, table=True):
    """One row per cluster. PK ``cluster_id`` (``cl_<12hex>``)."""

    __tablename__ = "cluster"  # type: ignore[assignment]
    __table_args__ = (
        # List all clusters for one (app, project, owner, kind) on each strategy
        # invocation; scope-first composite so clustering never mixes spaces.
        Index("ix_cluster_owner_kind", "app_id", "project_id", "owner_id", "kind"),
    )

    cluster_id: str = Field(primary_key=True)
    """Caller-minted opaque id (algo type carries it through verbatim).
    Format: ``cl_<12 hex chars>`` to mirror :func:`memcell._mint_memcell_id`."""

    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    """App / project scope segments. The aggregation key is
    ``(app_id, project_id, owner_id, kind)`` so a cluster set never spans
    two spaces."""

    owner_id: str = Field(index=True)
    """``user_id`` (kind=``user_memory``) or ``agent_id`` (kind=``agent_case``)."""

    owner_type: str
    """``"user"`` or ``"agent"`` — redundant with ``kind`` today but kept
    explicit so future kinds (e.g. tenant-level) can plug in without a
    schema change."""

    kind: str
    """``"user_memory"`` (episode-vector cluster, drives profile extraction)
    or ``"agent_case"`` (task_intent-vector cluster, drives skill extraction)."""

    centroid_blob: bytes = Field(sa_type=LargeBinary)
    """``np.float32`` centroid serialised via ``ndarray.tobytes()``. The
    repo round-trips bytes ↔ ``np.ndarray`` so callers see the algo type."""

    count: int
    """Number of members merged into this cluster (algo-maintained)."""

    last_ts_ms: int
    """Most recent member's timestamp as Unix epoch milliseconds — matches
    :attr:`everalgo.clustering.Cluster.last_ts` exactly so no lossy
    datetime ↔ int conversion is needed across the storage boundary."""

    preview_json: str
    """JSON-encoded ``list[str]`` — short text samples used by
    :func:`cluster_by_llm` ranking. Repo round-trips JSON ↔ list."""


class ClusterMember(BaseTable, table=True):
    """One row per (cluster, entity) link.

    Forward lookup (``cluster_id → list[member_id]``) is the algo-side
    ``Cluster.members`` view. Reverse lookup (``(member_type, member_id)
    → cluster_id``) is served by the composite index below — needed when
    a downstream consumer holds an entity id and wants its cluster.

    ``member_type`` is informational on the row (the parent ``Cluster.kind``
    already disambiguates), but kept explicit so the reverse index can be
    a single composite (member_type, member_id) without joining back.
    """

    __tablename__ = "cluster_member"  # type: ignore[assignment]
    __table_args__ = (Index("ix_cluster_member_reverse", "member_type", "member_id"),)

    cluster_id: str = Field(primary_key=True, foreign_key="cluster.cluster_id")
    """Parent cluster id."""

    member_id: str = Field(primary_key=True)
    """``memcell_id`` (member_type=``memcell``) or md entry_id
    (member_type=``case``) — the entity grouped into this cluster."""

    member_type: str
    """``"memcell"`` or ``"case"``. Echoes the parent cluster's ``kind``
    domain but kept on the row so the reverse index is self-contained."""

    added_ts: UtcDatetime = Field(sa_type=UtcDateTimeColumn)
    """When this entity was first attached to the cluster."""
