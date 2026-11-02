"""T12.1, TB5.1, TB6.1, T24.1 (b) on a real server."""

from __future__ import annotations

import contextlib
import json
import stat
from typing import Any

import httpx
import pytest

from .conftest import chat
from .observe import frontmatter, lance_ids, md_entries, rows, system_db, wait_converged
from .seed import TS, seed_world, write_episode

pytestmark = pytest.mark.acceptance


def _ids(srv: Any, owner: str) -> set[str]:
    return {
        e["entry_id"]
        for e in md_entries(srv.root)
        if e["owner"] == owner and e["kind"] == "episodes"
    }


def _write(srv: Any, owner: str, session: str, n: int, tag: str) -> set[str]:
    before = _ids(srv, owner)
    for i in range(n):
        assert srv.add(session, chat(owner, f"{tag}{i}", n=1)).status_code == 200
        assert srv.flush(session).status_code == 200
    return _ids(srv, owner) - before


def test_t12_1_ids_are_never_handed_out_twice(make_server: Any) -> None:
    """A12: after a session deletion, a whole-owner deletion, and a whole-owner
    deletion killed right after its rename (then a restart), new entries of
    the same day get new ids."""
    srv = make_server()
    srv.start()
    old = {o: _write(srv, o, f"{o}-s", 5, "apricot") for o in ("u1", "u2", "u3")}
    assert all(len(ids) == 5 for ids in old.values()), old

    assert srv.delete(session_id="u1-s").status_code == 200
    assert srv.delete(user_id="u2").status_code == 200
    srv.stop()
    srv.start(EVEROS_TEST_FAILPOINTS="delete_owner.after_rename")
    with contextlib.suppress(httpx.HTTPError):  # the process dies mid-request
        srv.post("/memory/delete", {"user_id": "u3"}, timeout=30)
    assert srv.wait_exit() == 137
    srv.env_extra.pop("EVEROS_TEST_FAILPOINTS")
    srv.start()  # startup sweep finishes u3's deletion
    assert not list((srv.root / ".tmp").glob("delete-*"))

    new = {o: _write(srv, o, f"{o}-t", 3, "quince") for o in ("u1", "u2", "u3")}
    for owner in ("u1", "u2", "u3"):
        assert len(new[owner]) == 3, (owner, new[owner])
        assert not new[owner] & old[owner], (owner, new[owner] & old[owner])


def test_tb5_1_cluster_follows_its_remaining_members(make_server: Any) -> None:
    """B5 / C13: a dangling member, then single ids, then empty."""
    srv = make_server()
    members = [
        write_episode(
            srv.root,
            "u1",
            f"2025-10-0{1 + i % 3}",
            session=f"s{i}",
            text=f"text {i}",
            ts=TS + i * 1000,
        )
        for i in range(7)
    ]
    import asyncio

    from everos.infra.persistence.factories import build_stores
    from everos.memory.models import Cluster
    from tests.fakes.storage import make_storage

    async def make_cluster() -> None:
        storage = make_storage(srv.root)
        await storage.create_schema()
        stores = build_stores(storage.root, storage.sqlite)
        await stores.cluster.upsert(
            _user(),
            "user_memory",
            Cluster(
                id="cl_t",
                centroid=(0.25,) * 4,
                count=7,
                last_ts=TS,
                preview=["p"],
                members=members,
            ),
            member_type="episode",
        )
        await storage.aclose()

    asyncio.run(make_cluster())
    db = system_db(srv.root)
    centroid = rows(db, "SELECT centroid_blob FROM cluster WHERE cluster_id='cl_t'")
    srv.start()

    assert srv.delete(session_id="s0").status_code == 200  # leaves member 0 dangling
    assert srv.delete_by_ids([f"u1_{members[6]}"]).status_code == 200

    count, last_ts, preview = rows(
        db,
        "SELECT count, last_ts_ms, preview_json FROM cluster WHERE cluster_id='cl_t'",
    )[0]
    left = sorted(
        r[0]
        for r in rows(
            db, "SELECT member_id FROM cluster_member WHERE cluster_id='cl_t'"
        )
    )
    assert left == sorted(members[1:6])
    assert (count, last_ts) == (5, TS + 5 * 1000)
    assert json.loads(preview) == [f"text {i}" for i in range(1, 6)]
    assert (
        rows(db, "SELECT centroid_blob FROM cluster WHERE cluster_id='cl_t'")
        == centroid
    )

    for m in members[1:6]:
        assert srv.delete_by_ids([f"u1_{m}"]).status_code == 200
    assert rows(db, "SELECT COUNT(*) FROM cluster WHERE cluster_id='cl_t'") == [(0,)]
    assert rows(db, "SELECT COUNT(*) FROM cluster_member WHERE cluster_id='cl_t'") == [
        (0,)
    ]


def _user() -> Any:
    from .seed import user

    return user("u1")


def test_tb6_1_two_rounds_of_merges_go_together(make_server: Any) -> None:
    """B6: M2 merged M1 and e3, M1 merged e1 and e2 (different days); deleting
    M2 by id removes all five and their marks."""
    srv = make_server()
    e1 = write_episode(srv.root, "u1", "2025-10-01", text="e1")
    e2 = write_episode(srv.root, "u1", "2025-10-02", text="e2")
    e3 = write_episode(srv.root, "u1", "2025-10-03", text="e3")
    m1 = write_episode(
        srv.root,
        "u1",
        "2025-10-04",
        session=None,
        parent_type="cluster",
        parent_id="cl",
        text="m1",
    )
    m2 = write_episode(
        srv.root,
        "u1",
        "2025-10-05",
        session=None,
        parent_type="cluster",
        parent_id="cl",
        text="m2",
    )
    import asyncio

    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.markdown import MarkdownWriter

    files = {e["entry_id"]: e["path"] for e in md_entries(srv.root)}
    w = MarkdownWriter(MemoryRoot(srv.root))
    for src, merged in ((e1, m1), (e2, m1), (m1, m2), (e3, m2)):
        asyncio.run(
            w.patch_frontmatter(
                srv.root / files[src], {"deprecated_entries": {src: merged}}
            )
        )
    srv.start()
    wait_converged(srv.base)

    assert srv.delete_by_ids([f"u1_{m2}"]).status_code == 200

    assert [e for e in md_entries(srv.root) if e["owner"] == "u1"] == []
    for path in set(files.values()):
        assert not frontmatter(srv.root, path).get("deprecated_entries")
    wait_converged(srv.base)
    assert {i for i in lance_ids(srv.root, "episode") if i.startswith("u1_")} == set()


def test_t24_1_b_unwritable_dir_is_500_and_the_resend_finishes(
    make_server: Any,
) -> None:
    """A24 (b): a rewrite that cannot write answers 500 "deletion incomplete";
    with the permission back, the same request finishes."""
    srv = make_server()
    seed_world(srv.root)
    srv.start()
    episodes = srv.root / "default_app/default_project/users/u1/episodes"
    expected = sum(1 for e in md_entries(srv.root) if e["session_id"] == "s1")
    mode = episodes.stat().st_mode
    episodes.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        first = srv.delete(session_id="s1")
    finally:
        episodes.chmod(mode)

    assert first.status_code == 500, first.text
    assert first.json()["error"] == {
        "code": "INTERNAL_ERROR",
        "message": "deletion incomplete; retry the same request",
    } | {k: v for k, v in first.json()["error"].items() if k not in ("code", "message")}
    second = srv.delete(session_id="s1")
    assert second.status_code == 200
    assert [e for e in md_entries(srv.root) if e["session_id"] == "s1"] == []
    assert expected > 0
