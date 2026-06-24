"""Repository for the ``cluster`` + ``cluster_member`` pair.

Bridges between the storage row shape and the algo-side
:class:`everalgo.clustering.Cluster` value object. Callers always work in
the algo type — this repo handles the centroid bytes round-trip, the
preview JSON round-trip, and the membership join so the algo's
``members: list[str]`` field is always fully populated on read. The
``last_ts`` field is stored as int milliseconds (matches the algo type
exactly) to keep the round-trip lossless across SQLite's tz-naive
``DateTime`` storage.

The single ``upsert_with_members`` write path is what every cluster
strategy invokes after a merge / new-cluster decision: it stamps the
``cluster`` row (UPSERT) and reconciles the ``cluster_member`` rows
(diff-then-insert; pre-existing members are kept, new members appended)
so calls are idempotent even if a strategy retries.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.component.utils.datetime import get_utc_now
from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import Cluster, ClusterMember

_CENTROID_DTYPE = np.float32


def mint_cluster_id() -> str:
    """Mint a fresh cluster id (mirrors ``_mint_memcell_id``: ``cl_<12hex>``)."""
    return f"cl_{uuid.uuid4().hex[:12]}"


class _ClusterRepo(RepoBase[Cluster]):
    """CRUD repository for the ``cluster`` + ``cluster_member`` table pair.

    Bridges between the SQLite row shape and the algo-side
    :class:`everalgo.clustering.Cluster` value object, handling centroid
    bytes round-trip, preview JSON serialisation, and membership joins.
    """

    model = Cluster

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    # ── Reads ──────────────────────────────────────────────────────────────

    async def get_with_members(self, cluster_id: str) -> AlgoCluster | None:
        """Fetch one cluster as a fully-hydrated algo value object.

        Returns ``None`` when no row matches ``cluster_id`` — downstream
        strategies that race the writer should treat this as a transient
        miss and let OME retry the run.
        """
        async with session_scope(self._factory) as s:
            row = await s.get(Cluster, cluster_id)
            if row is None:
                return None
            members_by_cluster = await _load_members_by_cluster(s, [cluster_id])
        return _row_to_algo(row, members_by_cluster.get(cluster_id, []))

    async def list_for_owner(
        self,
        owner_id: str,
        kind: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[AlgoCluster]:
        """All clusters for ``(app, project, owner, kind)``, as algo objects.

        Hot path for the cluster strategies (``cluster_by_geometry`` /
        ``cluster_by_llm`` need the full ``existing_clusters`` list). Each
        returned cluster carries its full ``members`` view, populated from
        the join with :class:`ClusterMember`. Scoping by (app, project)
        keeps one space's clusters from merging into another's.
        """
        async with session_scope(self._factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(Cluster)
                        .where(Cluster.app_id == app_id)
                        .where(Cluster.project_id == project_id)
                        .where(Cluster.owner_id == owner_id)
                        .where(Cluster.kind == kind)
                    )
                )
                .scalars()
                .all()
            )
            if not rows:
                return []
            ids = [r.cluster_id for r in rows]
            members_by_cluster = await _load_members_by_cluster(s, ids)
        return [
            _row_to_algo(row, members_by_cluster.get(row.cluster_id, []))
            for row in rows
        ]

    async def find_cluster_id_for_member(
        self,
        member_type: str,
        member_id: str,
    ) -> str | None:
        """Reverse lookup: ``(member_type, member_id) → cluster_id``.

        Returns ``None`` when the entity is not yet attached to any cluster.
        Backed by ``ix_cluster_member_reverse`` so it is O(log N).
        """
        async with session_scope(self._factory) as s:
            stmt = (
                select(ClusterMember.cluster_id)
                .where(ClusterMember.member_type == member_type)
                .where(ClusterMember.member_id == member_id)
                .limit(1)
            )
            return (await s.execute(stmt)).scalar_one_or_none()

    # ── Member-level CRUD ───────────────────────────────────────────────────

    async def remove_members(self, cluster_id: str, member_ids: set[str]) -> None:
        """Hard-delete specific member rows from cluster_member.

        Args:
            cluster_id: Target cluster identifier.
            member_ids: Set of member ids to remove; no-op when empty.
        """
        if not member_ids:
            return
        async with session_scope(self._factory) as s:
            await s.execute(
                delete(ClusterMember)
                .where(ClusterMember.cluster_id == cluster_id)
                .where(ClusterMember.member_id.in_(member_ids))
            )
            await s.commit()

    async def add_member(
        self, cluster_id: str, member_id: str, member_type: str
    ) -> None:
        """Add a single member to an existing cluster.

        Args:
            cluster_id: Target cluster identifier.
            member_id: Unique id of the member entity.
            member_type: Kind discriminator (e.g. ``"episode"``).
        """
        async with session_scope(self._factory) as s:
            s.add(
                ClusterMember(
                    cluster_id=cluster_id,
                    member_id=member_id,
                    member_type=member_type,
                    added_ts=get_utc_now(),
                )
            )
            await s.commit()

    async def update_metadata(
        self,
        cluster_id: str,
        *,
        centroid_blob: bytes,
        count: int,
        last_ts_ms: int,
        preview_json: str,
    ) -> None:
        """Update cluster metadata after member changes.

        Args:
            cluster_id: Target cluster identifier.
            centroid_blob: Serialised float32 centroid vector bytes.
            count: Updated member count.
            last_ts_ms: Latest member timestamp in epoch milliseconds.
            preview_json: JSON-encoded preview text list.
        """
        async with session_scope(self._factory) as s:
            await s.execute(
                update(Cluster)
                .where(Cluster.cluster_id == cluster_id)
                .values(
                    centroid_blob=centroid_blob,
                    count=count,
                    last_ts_ms=last_ts_ms,
                    preview_json=preview_json,
                )
            )
            await s.commit()

    # ── Lightweight queries ───────────────────────────────────────────────

    async def list_ids_and_member_counts(
        self,
        owner_id: str,
        kind: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[tuple[str, int]]:
        """Return ``(cluster_id, member_count)`` from actual member rows.

        Args:
            owner_id: Scope owner identifier.
            kind: Cluster kind discriminator.
            app_id: Application scope (default ``"default"``).
            project_id: Project scope (default ``"default"``).
        """
        async with session_scope(self._factory) as s:
            stmt = (
                select(
                    Cluster.cluster_id,
                    func.count(ClusterMember.member_id),
                )
                .join(
                    ClusterMember,
                    Cluster.cluster_id == ClusterMember.cluster_id,
                )
                .where(Cluster.owner_id == owner_id)
                .where(Cluster.kind == kind)
                .where(Cluster.app_id == app_id)
                .where(Cluster.project_id == project_id)
                .group_by(Cluster.cluster_id)
            )
            return list((await s.execute(stmt)).all())

    async def get_members_with_type(self, cluster_id: str) -> list[tuple[str, str]]:
        """Return ``(member_id, member_type)`` pairs for a cluster.

        Args:
            cluster_id: Target cluster identifier.
        """
        async with session_scope(self._factory) as s:
            stmt = (
                select(ClusterMember.member_id, ClusterMember.member_type)
                .where(ClusterMember.cluster_id == cluster_id)
                .order_by(ClusterMember.added_ts)
            )
            return list((await s.execute(stmt)).all())

    async def list_distinct_owners(
        self,
    ) -> list[tuple[str, str, str, str]]:
        """Return distinct ``(owner_id, owner_type, app_id, project_id)`` tuples.

        Used by the Reflection cron strategy to enumerate all scope
        combinations that may have clusters to reflect.
        """
        async with session_scope(self._factory) as s:
            stmt = select(
                Cluster.owner_id,
                Cluster.owner_type,
                Cluster.app_id,
                Cluster.project_id,
            ).distinct()
            return list((await s.execute(stmt)).all())

    # ── Write ──────────────────────────────────────────────────────────────

    async def upsert_with_members(
        self,
        algo_cluster: AlgoCluster,
        *,
        owner_id: str,
        owner_type: str,
        kind: str,
        member_type: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> None:
        """Persist one algo cluster snapshot + its membership rows.

        ``algo_cluster.id`` must be non-None (caller-minted via
        :func:`mint_cluster_id` for a brand-new cluster, or carried
        through from a merge return). ``algo_cluster.members`` is the
        full member list — the repo diffs against existing membership
        and inserts only the new rows so the call is idempotent under
        OME's at-least-once retry semantics.
        """
        cluster_id = algo_cluster.id
        if not cluster_id:
            raise ValueError(
                "upsert_with_members requires algo_cluster.id (mint via "
                "mint_cluster_id() before passing in)."
            )
        now = get_utc_now()
        centroid_blob = np.asarray(
            algo_cluster.centroid, dtype=_CENTROID_DTYPE
        ).tobytes()
        preview_json = json.dumps(list(algo_cluster.preview), ensure_ascii=False)

        async with session_scope(self._factory) as s:
            cluster_stmt = (
                sqlite_insert(Cluster)
                .values(
                    cluster_id=cluster_id,
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=owner_id,
                    owner_type=owner_type,
                    kind=kind,
                    centroid_blob=centroid_blob,
                    count=algo_cluster.count,
                    last_ts_ms=algo_cluster.last_ts,
                    preview_json=preview_json,
                )
                .on_conflict_do_update(
                    index_elements=["cluster_id"],
                    set_={
                        "centroid_blob": centroid_blob,
                        "count": algo_cluster.count,
                        "last_ts_ms": algo_cluster.last_ts,
                        "preview_json": preview_json,
                    },
                )
            )
            await s.execute(cluster_stmt)

            existing = set(
                (
                    await s.execute(
                        select(ClusterMember.member_id).where(
                            ClusterMember.cluster_id == cluster_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            new_member_rows = [
                ClusterMember(
                    cluster_id=cluster_id,
                    member_id=mid,
                    member_type=member_type,
                    added_ts=now,
                )
                for mid in algo_cluster.members
                if mid not in existing
            ]
            if new_member_rows:
                s.add_all(new_member_rows)
            await s.commit()


def _row_to_algo(row: Cluster, members: list[str]) -> AlgoCluster:
    centroid = np.frombuffer(row.centroid_blob, dtype=_CENTROID_DTYPE)
    preview = json.loads(row.preview_json) if row.preview_json else []
    return AlgoCluster(
        id=row.cluster_id,
        centroid=centroid,
        count=row.count,
        last_ts=row.last_ts_ms,
        preview=preview,
        members=list(members),
    )


async def _load_members_by_cluster(
    session: AsyncSession,
    cluster_ids: list[str],
) -> dict[str, list[str]]:
    """One query → ``{cluster_id: [member_id, ...]}`` (insertion order)."""
    stmt = (
        select(ClusterMember.cluster_id, ClusterMember.member_id)
        .where(ClusterMember.cluster_id.in_(cluster_ids))
        .order_by(ClusterMember.added_ts)
    )
    buckets: dict[str, list[str]] = {}
    for cluster_id, member_id in (await session.execute(stmt)).all():
        buckets.setdefault(cluster_id, []).append(member_id)
    return buckets


cluster_repo = _ClusterRepo()
