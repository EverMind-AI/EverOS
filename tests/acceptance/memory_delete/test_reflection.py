"""Deletions racing Reflection and clustering (Reflection enabled).

T22.1, T9.2, T15.1 (b), TB7.1 (four groups, main journey), T11.1 (b)(c).
Clusters are built through the real pipeline: two near-identical /adds of
one owner, clustered by the real embedding. A round whose cluster did not
form, or whose call was not held, does not count; each case needs 16 of 20
rounds, or it is "not manufactured".
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from .conftest import chat
from .observe import frontmatter, md_entries, rows, system_db, wait_idle
from .proxy import FACTS, Rule, llm_call, single_embedding

pytestmark = pytest.mark.acceptance

ROUNDS = 20
NEEDED = 16
MERGE = r"memory consolidation assistant"


@pytest.fixture
def refl(make_server: Any) -> Any:
    srv = make_server()
    (srv.root / "ome.toml").write_text(
        "[strategies.reflect_episodes]\nenabled = true\n"
    )
    srv.start()
    return srv


def _episodes(srv: Any, owner: str) -> list[dict[str, Any]]:
    return [
        e
        for e in md_entries(srv.root)
        if e["owner"] == owner and e["kind"] == "episodes"
    ]


def _members(srv: Any, owner: str) -> set[str]:
    found = rows(
        system_db(srv.root),
        "SELECT m.member_id FROM cluster_member m JOIN cluster c"
        " ON c.cluster_id = m.cluster_id WHERE c.owner_id = ?",
        owner,
    )
    return {m for (m,) in found}


def _clustered_pair(srv: Any, owner: str, token: str) -> list[str] | None:
    """Two near-identical conversations; their ids if clustered together."""
    for i in range(2):
        msg = {
            "sender_id": owner,
            "role": "user",
            "timestamp": int(time.time() * 1000) + i,
            "content": (
                f"Today I went to the {token} aquarium and watched the {token} sharks."
            ),
        }
        assert srv.add(f"{owner}-s{i}", [msg]).status_code == 200
        assert srv.flush(f"{owner}-s{i}").status_code == 200
    wait_idle(srv.base, srv.root)
    ids = [e["entry_id"] for e in _episodes(srv, owner)]
    cluster_rows = rows(
        system_db(srv.root),
        "SELECT m.cluster_id, m.member_id FROM cluster_member m JOIN cluster c"
        " ON c.cluster_id = m.cluster_id WHERE c.owner_id = ?",
        owner,
    )
    by_cluster: dict[str, set[str]] = {}
    for cid, mid in cluster_rows:
        by_cluster.setdefault(cid, set()).add(mid)
    if len(ids) == 2 and any(set(ids) <= m for m in by_cluster.values()):
        return ids
    return None


def _reflect(srv: Any) -> None:
    resp = srv.post("/ome/trigger", {"name": "reflect_episodes", "force": True})
    assert resp.status_code == 200, resp.text


def _reflect_in_background(srv: Any) -> None:
    """Trigger Reflection without waiting for it: the trigger answers only
    once its runs settle, by which time a held model call has timed out (60 s)
    and its retry has gone past the hold -- the deletion would come after."""
    threading.Thread(target=_reflect, args=(srv,), daemon=True).start()


def _window_open(srv: Any, rule: Rule) -> bool:
    """The held call is still held and is the only one of its kind since the
    rule was added: no timeout retry went past the hold before the deletion
    answered."""
    seen = sum(
        1
        for target, body in srv.proxy.bodies[rule.since :]
        if target == rule.target and re.search(rule.pattern, body, re.S)
    )
    return seen == 1 and srv.proxy.held(rule.target) >= 1


def _merged(srv: Any, owner: str) -> list[str]:
    return [
        e["entry_id"] for e in _episodes(srv, owner) if e["parent_type"] == "cluster"
    ]


def _gone(srv: Any, owner: str, target: str) -> None:
    """``target``, its facts and its cluster membership are nowhere."""
    entries = md_entries(srv.root)
    assert not [e for e in entries if e["owner"] == owner and e["entry_id"] == target]
    assert not [e for e in entries if e["owner"] == owner and e["parent_id"] == target]
    assert target not in _members(srv, owner)


def _watch(check: Callable[[], None], seconds: int = 20) -> None:
    for _ in range(seconds):
        check()
        time.sleep(1)


def _require(hits: int, label: str) -> None:
    print(f"{label}: window hit {hits}/{ROUNDS}")
    assert hits >= NEEDED, f"{label}: window hit {hits}/{ROUNDS}: not manufactured"


def _send_held(srv: Any, owner: str, token: str, rule: Rule, target: str) -> str | None:
    """One /add (+ flush) of ``owner`` whose ``target`` call gets held;
    returns the new episode's id once it is held, else None."""
    known = {e["entry_id"] for e in _episodes(srv, owner)}
    assert srv.add(f"{owner}-x", chat(owner, token, n=1)).status_code == 200
    threading.Thread(target=lambda: srv.flush(f"{owner}-x"), daemon=True).start()
    if not srv.proxy.wait_held(target, timeout=90):
        rule.release.set()
        srv.proxy.clear()
        return None
    new = [e["entry_id"] for e in _episodes(srv, owner) if e["entry_id"] not in known]
    return new[0] if new else None


def test_t22_1_merged_episode_gets_its_facts(refl: Any) -> None:
    """A22: with nothing deleted, merged episodes get their facts."""
    srv = refl
    owners = [f"t22u{r}" for r in range(3)]
    formed = sum(
        _clustered_pair(srv, o, f"dugong{r}q") is not None for r, o in enumerate(owners)
    )
    assert formed >= 1, "no cluster formed: not manufactured"
    _reflect(srv)
    time.sleep(2)
    wait_idle(srv.base, srv.root)
    entries = md_entries(srv.root)
    with_facts = sum(
        any(e["kind"] == ".atomic_facts" and e["parent_id"] == m for e in entries)
        for o in owners
        for m in _merged(srv, o)
    )
    assert with_facts >= 1
    srv.stop()
    assert "memory_write_skipped_source_gone" not in srv.log_text()


def test_t9_2_session_of_a_source_deleted_during_the_merge(refl: Any) -> None:
    """A9 (Refl) / spec 1 R20: the merge's model call is held; a source's
    session is deleted; the merge is abandoned."""
    srv = refl
    hits = 0
    for r in range(ROUNDS):
        owner, token = f"t92u{r}", f"narwhal{r}q"
        if _clustered_pair(srv, owner, token) is None:
            continue
        rule = srv.proxy.add(Rule("llm", llm_call(MERGE, token), "hold", times=1))
        _reflect_in_background(srv)
        if not srv.proxy.wait_held("llm", timeout=90):
            rule.release.set()
            srv.proxy.clear()
            continue
        assert srv.delete(session_id=f"{owner}-s0").status_code == 200
        if not _window_open(srv, rule):
            srv.proxy.clear()
            continue
        hits += 1
        rule.release.set()

        def check(o: str = owner) -> None:
            assert not _merged(srv, o), "a merge of a deleted source was written"
            assert not [e for e in md_entries(srv.root) if e["session_id"] == f"{o}-s0"]

        _watch(check)
        srv.proxy.clear()
    _require(hits, "T9.2")


def test_t15_1_b_finish_marks_only_sources_still_there(refl: Any) -> None:
    """A15 (b): M's fact extraction is held; a source's session is deleted;
    released, the finish succeeds and marks only the surviving source."""
    srv = refl
    ome = srv.root / ".index" / "sqlite" / "ome.db"
    hits = 0
    for r in range(ROUNDS):
        owner, token = f"t15u{r}", f"walrus{r}q"
        ids = _clustered_pair(srv, owner, token)
        if ids is None:
            continue
        rule = srv.proxy.add(Rule("llm", llm_call(FACTS, token), "hold", times=1))
        _reflect_in_background(srv)
        if not srv.proxy.wait_held("llm", timeout=90) or not _merged(srv, owner):
            rule.release.set()
            srv.proxy.clear()
            continue
        assert srv.delete(session_id=f"{owner}-s0").status_code == 200
        if not _window_open(srv, rule):
            srv.proxy.clear()
            continue
        hits += 1
        rule.release.set()
        wait_idle(srv.base, srv.root)
        (status,) = rows(
            ome,
            "SELECT status FROM run_record WHERE strategy_name='finish_reflection'"
            " ORDER BY started_at DESC LIMIT 1",
        )[0]
        assert status == "success", status
        marks: dict[str, str] = {}
        for e in _episodes(srv, owner):
            marks |= frontmatter(srv.root, e["path"]).get("deprecated_entries") or {}
        assert ids[0] not in marks, "a deleted source got a mark"
        assert ids[1] in marks, "the surviving source was not marked"
        srv.proxy.clear()
    _require(hits, "T15.1 (b)")


# B7 (main journey, Reflection on): four windows, 20 rounds each, one test
# (and one server) per window -- together they outlast a server's lifetime cap.


def test_tb7_1_1_fact_extraction_in_flight(refl: Any) -> None:
    """(1) The episode's fact extraction is in flight."""
    srv = refl
    hits = 0
    for r in range(ROUNDS):
        owner, token = f"tb71u{r}", f"bison{r}q"
        rule = srv.proxy.add(Rule("llm", llm_call(FACTS, token), "hold", times=1))
        target = _send_held(srv, owner, token, rule, "llm")
        if target is None:
            continue
        assert srv.delete_by_ids([f"{owner}_{target}"]).status_code == 200
        if not _window_open(srv, rule):
            srv.proxy.clear()
            continue
        hits += 1
        rule.release.set()
        _watch(lambda o=owner, t=target: _gone(srv, o, t))
        srv.proxy.clear()
    _require(hits, "TB7.1 (1)")


def test_tb7_1_2_clustering_in_its_embedding(refl: Any) -> None:
    """(2) The episode's clustering is in its embedding."""
    srv = refl
    hits = 0
    for r in range(ROUNDS):
        owner, token = f"tb72u{r}", f"tapir{r}q"
        rule = srv.proxy.add(Rule("emb", single_embedding(token), "hold", times=1))
        target = _send_held(srv, owner, token, rule, "emb")
        if target is None:
            continue
        assert srv.delete_by_ids([f"{owner}_{target}"]).status_code == 200
        if not _window_open(srv, rule):
            srv.proxy.clear()
            continue
        hits += 1
        rule.release.set()
        _watch(lambda o=owner, t=target: _gone(srv, o, t))
        srv.proxy.clear()
    _require(hits, "TB7.1 (2)")


def test_tb7_1_3_merge_in_flight_a_source_goes(refl: Any) -> None:
    """(3) The merge's model call is in flight; a source goes."""
    srv = refl
    hits = 0
    for r in range(ROUNDS):
        owner, token = f"tb73u{r}", f"okapi{r}q"
        ids = _clustered_pair(srv, owner, token)
        if ids is None:
            continue
        rule = srv.proxy.add(Rule("llm", llm_call(MERGE, token), "hold", times=1))
        _reflect_in_background(srv)
        if not srv.proxy.wait_held("llm", timeout=90):
            rule.release.set()
            srv.proxy.clear()
            continue
        assert srv.delete_by_ids([f"{owner}_{ids[0]}"]).status_code == 200
        if not _window_open(srv, rule):
            srv.proxy.clear()
            continue
        hits += 1
        rule.release.set()

        def check3(o: str = owner, t: str = ids[0]) -> None:
            _gone(srv, o, t)
            assert not _merged(srv, o), "a merge of a deleted source was written"

        _watch(check3)
        srv.proxy.clear()
    _require(hits, "TB7.1 (3)")


def test_tb7_1_4_merged_facts_in_flight_merged_goes(refl: Any) -> None:
    """(4) The merged episode's facts are in flight; M goes. The finish (which
    marks the sources) comes after those facts, so it never ran: the merge is
    undone and the sources stay, unmarked (spec 2 R6)."""
    srv = refl
    hits = 0
    for r in range(ROUNDS):
        owner, token = f"tb74u{r}", f"gecko{r}q"
        ids = _clustered_pair(srv, owner, token)
        if ids is None:
            continue
        rule = srv.proxy.add(Rule("llm", llm_call(FACTS, token), "hold", times=1))
        _reflect_in_background(srv)
        if not srv.proxy.wait_held("llm", timeout=90) or not _merged(srv, owner):
            rule.release.set()
            srv.proxy.clear()
            continue
        merged = _merged(srv, owner)[0]
        assert srv.delete_by_ids([f"{owner}_{merged}"]).status_code == 200
        if not _window_open(srv, rule):
            srv.proxy.clear()
            continue
        hits += 1
        rule.release.set()

        def check4(o: str = owner, m: str = merged, src: list[str] = ids) -> None:
            _gone(srv, o, m)
            left = {e["entry_id"]: e["path"] for e in _episodes(srv, o)}
            assert set(src) <= set(left), ("a source went with M", src, left)
            for path in set(left.values()):
                marks = frontmatter(srv.root, path).get("deprecated_entries") or {}
                assert m not in marks.values(), ("marked by a deleted M", path)

        _watch(check4)
        srv.proxy.clear()
    _require(hits, "TB7.1 (4)")


def test_t11_1_b_c_deletions_do_not_wait_on_held_embeddings(refl: Any) -> None:
    """A11 (b)(c): clustering stuck in its embedding, and a Reflection finish
    stuck in its embedding; deletions touching their owner answer within the
    index bound (15 s x tables + 1 s) instead of waiting."""
    srv = refl
    bound = 15 * 4 + 1
    ids = _clustered_pair(srv, "t11u", "manatee1q")
    assert ids is not None, "no cluster formed: not manufactured"
    rule = srv.proxy.add(Rule("emb", single_embedding("manatee2q"), "hold", times=1))
    assert _send_held(srv, "t11u", "manatee2q", rule, "emb") is not None
    started = time.monotonic()
    assert srv.delete_by_ids([f"t11u_{ids[0]}"]).status_code == 200
    assert srv.delete(user_id="t11u").status_code == 200
    assert time.monotonic() - started < bound
    assert _window_open(srv, rule), "clustering embedding not held through"
    rule.release.set()
    srv.proxy.clear()

    hit = False
    for attempt in range(3):
        owner, token = f"t11v{attempt}", f"manatee3{attempt}q"
        ids = _clustered_pair(srv, owner, token)
        if ids is None:
            continue
        finish = srv.proxy.add(Rule("emb", single_embedding(token), "hold", times=1))
        _reflect_in_background(srv)
        if not srv.proxy.wait_held("emb", timeout=120):
            finish.release.set()
            srv.proxy.clear()
            continue
        started = time.monotonic()
        assert srv.delete_by_ids([f"{owner}_{ids[1]}"]).status_code == 200
        took = time.monotonic() - started
        held_through = _window_open(srv, finish)
        finish.release.set()
        srv.proxy.clear()
        if not held_through:
            continue
        print(f"T11.1 (c): deletion answered in {took:.2f} s with the finish held")
        assert took < bound
        hit = True
        break
    assert hit, "T11.1 (c): finish embedding never held through: not manufactured"
