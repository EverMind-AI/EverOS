"""A file under ``groups/<g>/`` indexes as a group, never as a user.

Before group memory the cascade mapped every scope directory other than
``agents/`` to a user, so a ``groups/`` file would have indexed as a user with
an empty id (docs/group-memory.md, section 2). What these pin:

- the owner resolver maps ``groups`` to ``("group", <group_id>)`` and refuses
  a scope directory it does not know instead of guessing ``user``;
- a group episode written through the episode store lands under
  ``groups/<g>/episodes/`` with ``gep_`` ids, matches the ``group_episode``
  kind and projects into the ``episode`` table with ``owner_type = group``
  and its window columns filled;
- a group picture memory does the same in ``image_memory``, with its poster
  in ``sender_ids``;
- the attribution fields round-trip through markdown, and are written only
  when set (a one-to-one entry is unchanged).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.core.utils.datetime import today_with_timezone
from everos.infra.cascade._projections._common import resolve_owner
from everos.infra.cascade.registry import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import eq
from everos.infra.persistence.markdown import (
    MarkdownEpisodeStore,
    MarkdownImageMemoryStore,
    MarkdownReader,
)
from everos.memory.models import Episode, ImageMemory
from everos.memory.ports import NoSourceCheck, Scope
from tests.fakes import FakeEmbeddingProvider, handler_deps
from tests.fakes.storage import make_storage

_GROUP = Scope(
    app_id="default", project_id="default", owner_id="trip-team", owner_type="group"
)
_START = 1_772_442_120_000  # 2026-03-02T09:02:00Z
_END = 1_772_442_480_000  # 2026-03-02T09:08:00Z


class _Tokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


def test_the_resolver_knows_two_owner_directories() -> None:
    assert resolve_owner({"user_id": "u1"}, "a/p/users/u1/user.md") == ("u1", "user")
    assert resolve_owner(
        {"group_id": "trip-team"}, "a/p/groups/trip-team/episodes/e.md"
    ) == ("trip-team", "group")
    with pytest.raises(ValueError, match="no owner scope"):
        resolve_owner({"user_id": "u1"}, "a/p/teams/u1/episodes/e.md")
    # EverOS 2.0 has no agent memory: an older store's agent file has no owner.
    with pytest.raises(ValueError, match="no owner scope"):
        resolve_owner({"agent_id": "b"}, "a/p/agents/b/.cases/x.md")


def _episode() -> Episode:
    return Episode(
        owner_id="trip-team",
        episode="Chen moved the departure to April 30; Ava listed two flights.",
        summary="Departure moved",
        subject="Departure moved to April 30",
        timestamp=_END,
        session_id="trip-team-main",
        sender_ids=["u_chen", "u_betty", "bot_ava"],
        parent_id="mc_7f3a",
        subject_ids=["u_chen", "u_betty"],
        origin="mixed",
        source_start=_START,
    )


async def test_a_group_episode_indexes_with_the_group_as_owner(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    ref = await MarkdownEpisodeStore(root).append(
        _GROUP, _episode(), only_if=NoSourceCheck()
    )
    assert ref is not None
    day = today_with_timezone().strftime("%Y%m%d")
    assert ref.entry_id == f"gep_{day}_00000001"
    assert ref.md_path.startswith("default_app/default_project/groups/trip-team/")
    assert ref.id == f"trip-team_{ref.entry_id}"

    parsed = await MarkdownReader.read(tmp_path / ref.md_path)
    assert parsed.frontmatter["group_id"] == "trip-team"
    assert parsed.frontmatter["track"] == "group"
    inline = parsed.entries[0].as_structured().inline
    assert inline["source_start"] == "2026-03-02T09:02:00+00:00"
    assert inline["subject_ids"] == "[u_chen, u_betty]"
    assert inline["origin"] == "mixed"

    spec = match_kind(ref.md_path)
    assert spec is not None and spec.name == "group_episode"
    storage = make_storage(root)
    await storage.create_schema()
    try:
        handler = spec.handler_factory(
            handler_deps(
                memory_root=root,
                tokenizer=_Tokenizer(),
                index=storage.repos,
                embedding=FakeEmbeddingProvider(dim=1024),
            )
        )
        await handler.handle_added_or_modified(ref.md_path)
        rows = await storage.repos.episode.find_where(
            eq("md_path", ref.md_path), limit=10
        )
    finally:
        await storage.aclose()

    (row,) = rows
    assert row.owner_type == "group"
    assert row.owner_id == "trip-team"
    assert row.id == ref.id
    assert row.subject_ids == ["u_chen", "u_betty"]
    assert row.origin == "mixed"
    assert row.source_start == _dt.datetime(2026, 3, 2, 9, 2, tzinfo=_dt.UTC)
    assert row.source_end == _dt.datetime(2026, 3, 2, 9, 8, tzinfo=_dt.UTC)


async def test_a_group_picture_memory_carries_its_poster(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    image = ImageMemory(
        owner_id="trip-team",
        description="A boarding pass for the 08:10 flight",
        timestamp=_END,
        image_id="ab" * 32,
        session_id="trip-team-main",
        parent_id="mc_7f3a",
        sender_ids=["u_chen"],
        subject_ids=["u_chen"],
        origin="member",
    )
    (ref,) = await MarkdownImageMemoryStore(root).append_many(
        _GROUP, [image], only_if=NoSourceCheck()
    )
    assert ref.entry_id.startswith("gim_")
    spec = match_kind(ref.md_path)
    assert spec is not None and spec.name == "group_image_memory"
    storage = make_storage(root)
    await storage.create_schema()
    try:
        handler = spec.handler_factory(
            handler_deps(
                memory_root=root,
                tokenizer=_Tokenizer(),
                index=storage.repos,
                embedding=None,
            )
        )
        await handler.handle_added_or_modified(ref.md_path)
        (row,) = await storage.repos.image_memory.find_where(
            eq("md_path", ref.md_path), limit=10
        )
    finally:
        await storage.aclose()

    assert (row.owner_type, row.owner_id) == ("group", "trip-team")
    assert row.sender_ids == ["u_chen"]
    assert row.source_start == row.source_end == row.timestamp
