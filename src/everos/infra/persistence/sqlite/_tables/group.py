"""``group_meta`` / ``group_member`` / ``session_binding`` -- the roster mirror.

``group_message`` sits beside them but mirrors nothing: it is the group's
dedupe ledger (docs/group-memory.md, section 5.2), system state like the
memcells and the buffer -- one row per message the group took, never derived
from markdown, so ``everos cascade rebuild`` leaves it as it is.

``groups/<group_id>/group.md`` is the truth (docs/group-memory.md, section
6.1); these three tables are what every read asks, written in the same locked
operation as the markdown so that a member removed from a group loses access
at once (section 6.2). ``everos cascade rebuild`` re-derives all three from
``group.md``.

Times are epoch milliseconds, the API's unit, so a membership window compiles
into an index predicate without a conversion. ``created_at`` of
``group_meta`` is the group's creation time, not the row's: the mirror is
rebuildable, and a rebuilt row must read the same.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, Index
from sqlmodel import Field

from everos.infra.persistence.sqlite._base.base import BaseTable


class GroupMeta(BaseTable, table=True):
    """One group. PK ``(app_id, project_id, group_id)``."""

    __tablename__ = "group_meta"  # type: ignore[assignment]

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    group_id: str = Field(primary_key=True)
    name: str = ""
    description: str = ""
    group_created_at: int = Field(sa_column=Column(BigInteger, nullable=False))
    """Epoch ms the group was created (its first message for an implicit one)."""
    created_by: str | None = None
    history_visible_from: str = "join"
    deliverer_id: str | None = None
    """The group's deliverer (``group.md``'s ``deliverer_id``); ``NULL`` for
    none. Added after the first release (``schema._ADDED_COLUMNS``)."""
    extra_json: str = "{}"
    md_path: str


class GroupMember(BaseTable, table=True):
    """One membership interval.

    PK ``(app_id, project_id, group_id, member_id, seq)``.
    """

    __tablename__ = "group_member"  # type: ignore[assignment]
    __table_args__ = (
        # "Which groups is this member in" -- the project roll-up's question.
        Index("ix_group_member_member", "app_id", "project_id", "member_id"),
    )

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    group_id: str = Field(primary_key=True)
    member_id: str = Field(primary_key=True)
    seq: int = Field(primary_key=True)
    """1-based count of this member's intervals in the group."""
    kind: str
    display_name: str
    aliases_json: str = "[]"
    role: str
    title: str = ""
    joined_at: int = Field(sa_column=Column(BigInteger, nullable=False))
    left_at: int | None = Field(default=None, sa_column=Column(BigInteger))
    source: str = "api"


class GroupMessage(BaseTable, table=True):
    """One message a group holds: its dedupe key and the session holding it.

    PK ``(app_id, project_id, group_id, key)`` -- the lookup a batch makes,
    one ``IN`` over its keys. ``key`` is ``id:<message_id>`` for a message
    the client gave an id, else ``fp:<sender_id>:<epoch ms>:<sha256 of the
    content as sent>`` (:func:`everos.memory.groups.message_key`). A row is
    written once its message is stored, moves to a later session that takes
    the message again after a ``/memory/delete`` released it, and is never
    removed by a forget.
    """

    __tablename__ = "group_message"  # type: ignore[assignment]

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    group_id: str = Field(primary_key=True)
    key: str = Field(primary_key=True)
    session_id: str


class SessionBinding(BaseTable, table=True):
    """A session's group. PK ``(app_id, project_id, session_id)``; never rebound."""

    __tablename__ = "session_binding"  # type: ignore[assignment]
    __table_args__ = (
        Index("ix_session_binding_group", "app_id", "project_id", "group_id"),
    )

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    group_id: str
    bound_at: int = Field(sa_column=Column(BigInteger, nullable=False))
