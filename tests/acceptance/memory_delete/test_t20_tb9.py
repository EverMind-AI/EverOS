"""T20.1 <- A20 and TB9.1 <- B9: a deletion killed at each crash point, then
resent after a restart, ends exactly where an uninterrupted one does."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from .observe import frontmatter, lance_ids, logical_dump, md_tree_hash, wait_converged
from .seed import TS, seed_two_rounds, write_episode

pytestmark = pytest.mark.acceptance

_DAYS = 200
_CONVERGE_S = 900
"""Indexing 800 seeded episodes with a real embedding model takes minutes."""


def _seed_big_session(root: Path) -> None:
    """Two users, each with an episode of session S in 200 day files (plus an
    unrelated one per file), and S's memcell naming both."""
    from everalgo.types import ChatMessage, MemCell

    from everos.infra.persistence.factories import build_stores
    from everos.memory.models import MemcellRecord
    from everos.memory.ports import SessionKey
    from tests.fakes.storage import make_storage

    start = dt.date(2025, 1, 1)
    for i in range(_DAYS):
        day = (start + dt.timedelta(days=i)).isoformat()
        for owner in ("u1", "u2"):
            write_episode(
                root, owner, day, session="S", parent_id="mc_S", text=f"S {i}"
            )
            write_episode(
                root, owner, day, session="K", parent_id="mc_K", text=f"K {i}"
            )

    async def memcells() -> None:
        storage = make_storage(root)
        await storage.create_schema()
        stores = build_stores(storage.root, storage.sqlite)
        for session in ("S", "K"):
            await stores.memcell.insert_many(
                SessionKey(app_id="default", project_id="default", session_id=session),
                [
                    MemcellRecord(
                        memcell_id=f"mc_{session}",
                        cell=MemCell(
                            items=[
                                ChatMessage(
                                    id="m",
                                    role="user",
                                    content="x",
                                    timestamp=TS,
                                    sender_id="u1",
                                )
                            ],
                            timestamp=TS,
                        ),
                        raw_type="conversation",
                        message_ids=[],
                        sender_ids=["u1", "u2"],
                    )
                ],
            )
        await storage.aclose()

    asyncio.run(memcells())


def _state(srv: Any) -> tuple[str, Any, set[str]]:
    wait_converged(srv.base, timeout=_CONVERGE_S)
    dump = logical_dump(srv.root)
    dump.pop("memorize_inbox", None)
    return md_tree_hash(srv.root), dump, lance_ids(srv.root, "episode")


def _crash_then_resend(
    srv: Any, point: str, request: tuple[str, dict[str, Any]]
) -> None:
    path, body = request
    srv.start(EVEROS_TEST_FAILPOINTS=point)
    wait_converged(srv.base, timeout=_CONVERGE_S)
    with contextlib.suppress(httpx.HTTPError):
        srv.post(path, body, timeout=120)
    assert srv.wait_exit() == 137, point
    srv.env_extra.pop("EVEROS_TEST_FAILPOINTS")
    srv.start()


@pytest.mark.parametrize(
    "point",
    ["delete.after_sqlite", "delete.after_first_rewrite", "delete.before_index_erase"],
)
def test_t20_1_session_deletion_resent_after_a_crash(
    make_server: Any, point: str
) -> None:
    ref = make_server("reference")
    _seed_big_session(ref.root)
    test = make_server("crashed")
    shutil.copytree(ref.root, test.root, dirs_exist_ok=True)
    request = ("/memory/delete", {"session_id": "S"})

    ref.start()
    wait_converged(ref.base, timeout=_CONVERGE_S)
    assert ref.post(*request).status_code == 200
    expected = _state(ref)

    _crash_then_resend(test, point, request)
    assert test.post(*request).status_code == 200

    assert _state(test) == expected


def test_t20_1_index_catches_up_without_a_resend(make_server: Any) -> None:
    """Killed just before the index step and not resent: the restarted
    server's cascade removes the rows of what md no longer holds."""
    srv = make_server()
    _seed_big_session(srv.root)
    _crash_then_resend(
        srv, "delete.before_index_erase", ("/memory/delete", {"session_id": "S"})
    )
    time.sleep(60)
    md_ids = {
        f"{p.parent.parent.name}_{e}"
        for p in srv.root.rglob("episodes/*.md")
        for e in re.findall(r"<!-- entry:(\S+) -->", p.read_text())
    }
    assert lance_ids(srv.root, "episode") <= md_ids


@pytest.mark.parametrize(
    "point", ["delete_owner.after_rename", "delete_owner.after_restore"]
)
def test_t20_1_owner_deletion_finished_by_the_restart(
    make_server: Any, point: str
) -> None:
    from everos.core.utils.datetime import today_with_timezone

    srv = make_server()
    _seed_big_session(srv.root)
    today = today_with_timezone().isoformat()
    for i in range(3):  # today's file stands at id 3 before the deletion
        write_episode(srv.root, "u1", today, session="T", text=f"today {i}")
    _crash_then_resend(srv, point, ("/memory/delete", {"user_id": "u1"}))

    assert not list((srv.root / ".tmp").glob("delete-*"))
    u1 = srv.root / "default_app/default_project/users/u1"
    left = sorted(f.relative_to(u1).as_posix() for f in u1.rglob("*.md"))
    assert left == [f"episodes/episode-{today}.md"], left  # only today's, kept
    kept = frontmatter(srv.root, f"default_app/default_project/users/u1/{left[0]}")
    assert kept["entry_count"] == 3, kept
    assert "<!-- entry:" not in (u1 / left[0]).read_text()


@pytest.mark.parametrize(
    "point", ["by_ids.after_step1", "by_ids.after_step2", "by_ids.after_step4"]
)
def test_tb9_1_by_id_deletion_resent_after_a_crash(
    make_server: Any, point: str
) -> None:
    ref = make_server("reference")
    m2 = seed_two_rounds(ref.root)["m2"]  # with M2's cluster
    test = make_server("crashed")
    shutil.copytree(ref.root, test.root, dirs_exist_ok=True)
    request = (
        "/memory/delete_by_ids",
        {"memory_type": "episode", "memory_ids": [f"u1_{m2}"]},
    )

    ref.start()
    wait_converged(ref.base, timeout=_CONVERGE_S)
    assert _state(ref)[1]["cluster_member"], "positive control: M2 is clustered"
    assert ref.post(*request).status_code == 200
    expected = _state(ref)
    assert not expected[1]["cluster_member"]
    assert not {i for i in expected[2] if i.startswith("u1_")}

    _crash_then_resend(test, point, request)
    assert test.post(*request).status_code == 200

    assert _state(test) == expected
