"""``delete_all`` empties a LanceDB index table and keeps it usable."""

from __future__ import annotations

import datetime as dt

from everos.infra.persistence.records import Episode
from tests.fakes import TEST_DIM
from tests.fakes.storage import Storage


def _episode(i: int, app: str) -> Episode:
    return Episode(
        id=f"u1_ep{i}",
        entry_id=f"ep{i}",
        owner_id="u1",
        owner_type="user",
        app_id=app,
        timestamp=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        parent_id="mc1",
        sender_ids=["u1"],
        episode=f"episode {i}",
        episode_tokens=f"episode {i}",
        md_path=f"{app}/default_project/users/u1/episodes/day.md",
        content_sha256="a",
        vector=[1.0] + [0.0] * (TEST_DIM - 1),
    )


async def test_delete_all_empties_table_keeps_schema(storage: Storage) -> None:
    repo = storage.repos.episode
    await repo.upsert([_episode(i, app) for i, app in enumerate(("a1", "a2", "a3"))])
    assert await repo.count() == 3
    await repo.delete_all()
    assert await repo.count() == 0
    await repo.upsert([_episode(9, "a1")])  # same table, same schema
    assert await repo.count() == 1
