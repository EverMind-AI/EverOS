"""Repository for the ``reflection_report`` table."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import ReflectionReport


class _ReflectionReportRepo(RepoBase[ReflectionReport]):
    """CRUD repository for the ``reflection_report`` audit table.

    Provides creation, latest-by-cluster lookup, and reflected-cluster
    enumeration used by the Reflection orchestrator and cron strategy.
    """

    model = ReflectionReport

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def create(self, report: ReflectionReport) -> None:
        """Persist a new reflection report row.

        Args:
            report: Fully populated ReflectionReport instance.
        """
        async with session_scope(self._factory) as s:
            s.add(report)
            await s.commit()

    async def get_latest_for_cluster(self, cluster_id: str) -> ReflectionReport | None:
        """Most recent completed report for a cluster, or ``None``.

        Args:
            cluster_id: Cluster identifier to look up.
        """
        async with session_scope(self._factory) as s:
            stmt = (
                select(ReflectionReport)
                .where(ReflectionReport.cluster_id == cluster_id)
                .where(ReflectionReport.status == "completed")
                .order_by(ReflectionReport.created_at.desc())
                .limit(1)
            )
            return (await s.execute(stmt)).scalar_one_or_none()

    async def list_reflected_cluster_ids(
        self,
        owner_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> set[str]:
        """Distinct cluster ids that have at least one completed report.

        Args:
            owner_id: Scope owner identifier.
            app_id: Application scope (default ``"default"``).
            project_id: Project scope (default ``"default"``).
        """
        async with session_scope(self._factory) as s:
            stmt = (
                select(ReflectionReport.cluster_id)
                .where(ReflectionReport.owner_id == owner_id)
                .where(ReflectionReport.app_id == app_id)
                .where(ReflectionReport.project_id == project_id)
                .where(ReflectionReport.status == "completed")
                .distinct()
            )
            rows = (await s.execute(stmt)).scalars().all()
            return set(rows)


reflection_report_repo = _ReflectionReportRepo()
