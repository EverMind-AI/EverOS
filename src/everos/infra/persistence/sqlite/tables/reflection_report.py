"""ReflectionReport — audit record for each Reflection operation."""

from __future__ import annotations

from sqlmodel import Field

from everos.component.utils.datetime import UtcDatetime, get_utc_now
from everos.core.persistence.sqlite import BaseTable


class ReflectionReport(BaseTable, table=True):
    """One row per completed Reflection merge operation.

    Attributes:
        id: Primary key for this report.
        cluster_id: Cluster that was reflected.
        owner_id: Scope owner (user or agent id).
        app_id: Application scope, default ``"default"``.
        project_id: Project scope, default ``"default"``.
        mode: Reflection strategy mode (``"init"`` or ``"update"``).
        source_members: JSON-encoded list of member ids consumed.
        source_count: Number of source members consumed.
        merged_entry_id: Entry id of the merged output written to storage.
        deprecated_fact_count: Number of facts deprecated during merge.
        status: Completion status, default ``"completed"``.
        created_at: UTC timestamp when the report was created.
    """

    __tablename__ = "reflection_report"

    id: str = Field(primary_key=True)
    cluster_id: str = Field(index=True)
    owner_id: str = Field(index=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    mode: str
    source_members: str
    source_count: int
    merged_entry_id: str
    deprecated_fact_count: int
    status: str = Field(default="completed")
    created_at: UtcDatetime = Field(default_factory=get_utc_now)
