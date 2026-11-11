"""T21.1, T22.1, T23.1: what is in flight around a purge stays out of S."""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any

import httpx
import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule
from tests.acceptance.memory_delete.seed import write_episode

from .journey import finish, purge
from .observe import (
    INDEX_TABLES,
    index_ids,
    release,
    rows,
    scope_counts,
    system_db,
    wait_converged,
    wait_idle,
)
from .seed import AGENT, scope_token

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}
WHERE = {"app_id": "a1", "project_id": "p1"}


def _until_step6_done(srv: Any, job_id: str, timeout: float = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = srv.job(job_id).json()["data"]
        if {s["store"]: s["state"] for s in data["steps"]}["md_change_state"] == "done":
            return data
        time.sleep(0.2)
    raise TimeoutError("the purge never finished its first round")


def _lance_rows_with_app(root: Any, app: str) -> int:
    import lancedb

    db = lancedb.connect(str(root / ".index" / "lancedb"))
    total = 0
    for table in INDEX_TABLES:
        try:
            data = db.open_table(table).to_arrow().select(["app_id"]).to_pylist()
        except (ValueError, FileNotFoundError):
            continue
        total += sum(1 for r in data if r["app_id"] == app)
    return total


def test_t21_1_trash_is_never_indexed(make_world: Any) -> None:
    world = make_world(
        "T21.1",
        EVEROS_CASCADE__SCAN_INTERVAL_SECONDS="1",
        EVEROS_TEST_FAILPOINTS="purge.after_step2b=pause",
    )
    srv, root = world.srv, world.srv.root
    job_id, _ = purge(srv, **S)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not list((root / ".tmp").glob("purge-*")):
        time.sleep(0.2)
    assert list((root / ".tmp").glob("purge-*")), "never paused after step 2b"
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    write_episode(root, "u1", yesterday, app="a2", project="p1", text="hand edit")
    new_path = f"a2/p1/users/u1/episodes/episode-{yesterday}.md"
    time.sleep(5)
    registered = rows(
        system_db(root),
        "SELECT md_path FROM md_change_state WHERE md_path = ?",
        new_path,
    )
    assert registered, "no scan ran in the window: the check below proves nothing"
    release(srv.fp_dir, "purge.after_step2b")
    assert finish(srv, job_id)["status"] == "succeeded"
    wait_idle(srv.base, root, timeout=300)
    trash_rows = rows(
        system_db(root),
        "SELECT md_path FROM md_change_state WHERE md_path LIKE '.tmp/%'",
    )
    knowledge = rows(
        system_db(root), "SELECT COUNT(*) FROM knowledge_documents WHERE app_id='.tmp'"
    )
    world.ev.json("trash_rows", trash_rows)
    assert trash_rows == [] and knowledge == [(0,)]
    assert _lance_rows_with_app(root, ".tmp") == 0


def test_t22_1_sync_add_in_flight_is_refused(make_world: Any, purge_proxy: Any) -> None:
    world = make_world("T22.1", EVEROS_PURGE__SETTLE_WAIT_SECONDS="60")
    srv, root = world.srv, world.srv.root
    token = scope_token("a1", "p1")
    word = f"{token}late"
    hold = purge_proxy.add(
        Rule(target="llm", pattern=f'(?=.*"_EpisodeReply")(?=.*{word})', mode="hold")
    )
    responses: dict[str, httpx.Response] = {}

    def add_and_flush() -> None:
        body = {"session_id": "late", **WHERE}
        responses["add"] = srv.post(
            "/memory/add", {**body, "messages": chat("u1", word, assistant=AGENT)}
        )
        responses["flush"] = srv.post("/memory/flush", body)

    client = threading.Thread(target=add_and_flush)
    client.start()
    assert purge_proxy.wait_held("llm", timeout=60)
    job_id, _ = purge(srv, **S)
    first = _until_step6_done(srv, job_id)
    first_sqlite = {s["store"]: s["deleted"] for s in first["steps"]}["sqlite"]
    hold.release.set()
    client.join(timeout=120)
    job = finish(srv, job_id, timeout=180)
    world.ev.json(
        "responses", {k: (r.status_code, r.text[:400]) for k, r in responses.items()}
    )
    world.ev.json("job", job)
    refused = [r for r in responses.values() if r.status_code == 409]
    assert refused and refused[0].json()["error"]["code"] == "PURGE_IN_PROGRESS"
    counts = scope_counts(root, "a1", "p1")
    for key in (
        "sqlite.memcell",
        "sqlite.unprocessed_buffer",
        "sqlite.conversation_status",
        "md.episode",
    ):
        assert counts[key] == 0, (key, counts)
    final_sqlite = {s["store"]: s["deleted"] for s in job["steps"]}["sqlite"]
    world.ev.json(
        "sqlite_deleted", {"first_round": first_sqlite, "final": final_sqlite}
    )
    assert final_sqlite >= first_sqlite


def test_t23_1_projection_in_flight_is_cleared(
    make_world: Any, purge_proxy: Any
) -> None:
    """A projection that does not read the fence (skill / profile / knowledge)
    in flight across the first round: step 7 waits for it, step 8 clears
    what it writes. The profile projection computes no vector, so the skill
    projection, which embeds, is the one held."""
    world = make_world("T23.1")
    srv, root = world.srv, world.srv.root
    hold = purge_proxy.add(Rule(target="emb", pattern="zqskillonly", mode="hold"))
    (skill_md,) = sorted((root / "a1" / "p1" / "agents").rglob("SKILL.md"))
    text = skill_md.read_text(encoding="utf-8")
    assert "naive" in text
    skill_md.write_text(text.replace("naive", "naive zqskillonly"), encoding="utf-8")
    assert purge_proxy.wait_held("emb", timeout=120)
    job_id, _ = purge(srv, **S)
    _until_step6_done(srv, job_id)
    hold.release.set()
    assert finish(srv, job_id, timeout=180)["status"] == "succeeded"
    wait_converged(srv.base, timeout=240)
    left = index_ids(root, "agent_skill", "a1", "p1")
    world.ev.json("agent_skill_rows", sorted(left))
    assert left == set()
