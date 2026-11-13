"""Requests that race: what must hold when they arrive together.

The real server over the scripted model (``tests.helpers.faults``), driven
through its routes with ``asyncio.gather``: many adds to one session, adds to
one group from several sessions, a flush among adds, a forget among adds,
flushes and searches, an undo racing a new forget, two changes of one record,
searches during writes, OME profile updates during writes, and the session
lock against a slow model. Afterwards every marker sent is in exactly one
live episode (or, for a forget, marked forgotten -- never both, never lost),
markdown and the index agree, and no staging file is left.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.helpers.faults import (
    T0,
    FaultServer,
    assert_exactly_once,
    line,
    markers,
)

_ENTRY = re.compile(r"<!-- entry:(\S+) -->(.*?)<!-- /entry:\1 -->", re.S)


def _all_episode_markers(root: Path, owner: str) -> Counter[str]:
    """Markers of every episode entry in markdown, live or marked."""
    found: Counter[str] = Counter()
    for path in root.rglob("episode-*.md"):
        if f"/{owner}/" not in path.as_posix():
            continue
        for _id, text in _ENTRY.findall(path.read_text("utf-8")):
            found.update(set(markers(text)))
    return found


async def _group(srv: FaultServer, group: str, members: list[str]) -> None:
    resp = await srv.client.put(
        f"/api/v1/groups/{group}",
        json={
            "asker_id": members[0],
            "roster": [
                {"op": "join", "member_id": m, "at": T0, "role": r}
                for m, r in zip(
                    members, ["owner", *["member"] * (len(members) - 1)], strict=True
                )
            ],
        },
    )
    assert resp.status_code == 200, resp.text


async def _settled_and_clean(srv: FaultServer) -> None:
    await srv.settle()
    assert srv.staging() == []
    await srv.consistent()


def _ok(responses: list[httpx.Response]) -> None:
    bad = [(r.status_code, r.text[:200]) for r in responses if r.status_code != 200]
    assert bad == [], bad


# ── adds ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cut_on_add", [False, True], ids=["buffered", "cut-each"])
async def test_many_adds_to_one_session(tmp_path: Path, cut_on_add: bool) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = cut_on_add
        results = await asyncio.gather(
            *(srv.say("s1", "u1", f"C{n:02d}", 3, at=n * 10_000) for n in range(16))
        )
        _ok([r for r, _ in results])
        _ok([await srv.flush("s1")])
        assert_exactly_once(srv.episodes("u1"), [m for _, s in results for m in s])
        await _settled_and_clean(srv)
        assert_exactly_once(srv.episodes("u1"), [m for _, s in results for m in s])


async def test_adds_to_one_group_from_several_sessions(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        await _group(srv, "trip", ["u1", "u2", "u3"])
        srv.upstream.cut_on_add = True
        results = await asyncio.gather(
            *(
                srv.say(f"s{n % 3}", f"u{n % 3 + 1}", f"G{n:02d}", 2, group="trip",
                        at=n * 10_000)
                for n in range(12)
            )
        )  # fmt: skip
        _ok([r for r, _ in results])
        _ok([await srv.flush(f"s{n}") for n in range(3)])
        await _settled_and_clean(srv)
        assert_exactly_once(srv.episodes("trip"), [m for _, s in results for m in s])


async def test_the_same_group_batch_sent_twice_at_once_is_taken_once(
    tmp_path: Path,
) -> None:
    """A client retrying a slow add: two copies of one batch, with message ids."""
    async with FaultServer.open(tmp_path / "root") as srv:
        await _group(srv, "trip", ["u1", "u2", "u3"])
        srv.upstream.cut_on_add = True
        batch = [
            line("u1", f"note ZQD{i:02d}", i * 1000, message_id=f"m{i}")
            for i in range(3)
        ]
        first, second = await asyncio.gather(
            srv.add("s1", *batch, group="trip"), srv.add("s1", *batch, group="trip")
        )
        _ok([first, second])
        _ok([await srv.flush("s1")])
        await _settled_and_clean(srv)
        assert_exactly_once(srv.episodes("trip"), ["ZQD00", "ZQD01", "ZQD02"])


async def test_flushes_racing_adds(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        calls: list[Any] = []
        for n in range(10):
            calls.append(srv.say("s1", "u1", f"F{n:02d}", 2, at=n * 10_000))
            if n % 3 == 0:
                calls.append(srv.flush("s1"))
        results = await asyncio.gather(*calls)
        responses = [r[0] if isinstance(r, tuple) else r for r in results]
        _ok(responses)
        _ok([await srv.flush("s1")])
        sent = [m for r in results if isinstance(r, tuple) for m in r[1]]
        await _settled_and_clean(srv)
        assert_exactly_once(srv.episodes("u1"), sent)


# ── forgets and undos ───────────────────────────────────────────────────────


async def _forget(
    srv: FaultServer, user: str, target: dict[str, Any]
) -> httpx.Response:
    return await srv.post(
        "/memory/forget", {"user_id": user, "target": target, "dry_run": False}
    )


async def test_a_session_forget_racing_adds_flushes_and_searches(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        resp, old = await srv.say("s1", "u1", "O", 3)
        _ok([resp])
        await srv.settle()
        calls: list[Any] = [_forget(srv, "u1", {"kind": "session", "session_id": "s1"})]
        calls += [
            srv.say("s1", "u1", f"N{n}", 2, at=(n + 1) * 60_000) for n in range(6)
        ]
        calls += [
            srv.say("s2", "u1", f"M{n}", 2, at=(n + 1) * 60_000) for n in range(4)
        ]
        calls += [srv.flush("s1"), srv.flush("s2")]
        calls += [srv.search("notes", user_id="u1") for _ in range(6)]
        results = await asyncio.gather(*calls)
        responses = [r[0] if isinstance(r, tuple) else r for r in results]
        statuses = Counter(r.status_code for r in responses)
        assert set(statuses) <= {200, 503}, [r.text[:200] for r in responses]
        forgot = responses[0]
        sent = old + [m for r in results if isinstance(r, tuple) for m in r[1]]
        await srv.settle()
        live = srv.episodes("u1")
        everything = _all_episode_markers(srv.root, "u1")
        # Never lost (live or marked), never twice live.
        lost = sorted(set(sent) - set(everything))
        doubled = sorted(m for m, n in live.items() if n > 1)
        assert not lost and not doubled, (lost, doubled)
        if forgot.status_code == 200:
            assert not set(old) & set(live), live
        assert srv.staging() == []
        await srv.consistent()


async def test_an_undo_racing_a_new_forget_of_the_same_session(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        resp, first = await srv.say("s1", "u1", "U", 3)
        resp2, other = await srv.say("s2", "u1", "V", 3, at=60_000)
        _ok([resp, resp2])
        await srv.settle()
        target = {"kind": "session", "session_id": "s1"}
        done = await _forget(srv, "u1", target)
        _ok([done])
        forget_id = done.json()["data"]["forget_id"]
        undo, again, other_forget = await asyncio.gather(
            srv.post("/memory/forget/undo", {"user_id": "u1", "forget_id": forget_id}),
            _forget(srv, "u1", target),
            _forget(srv, "u1", {"kind": "session", "session_id": "s2"}),
        )
        _ok([undo, again, other_forget])
        await srv.settle()
        live = srv.episodes("u1")
        # Whichever ran first, the second forget of s1 stands at the end.
        assert not (set(first) | set(other)) & set(live), live
        await srv.consistent()
        # And undoing both brings everything back, once.
        for fid in (
            again.json()["data"]["forget_id"],
            other_forget.json()["data"]["forget_id"],
        ):
            _ok(
                [
                    await srv.post(
                        "/memory/forget/undo", {"user_id": "u1", "forget_id": fid}
                    )
                ]
            )
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), [*first, *other])
        await srv.consistent()


async def test_two_changes_of_one_record_one_wins_the_other_is_a_conflict(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        resp, _ = await srv.say("s1", "u1", "R", 2)
        _ok([resp, await srv.flush("s1")])
        await srv.settle()
        listed = await srv.post(
            "/memory/get", {"user_id": "u1", "memory_type": "episode"}
        )
        (record,) = listed.json()["data"]["episodes"]
        changes = await asyncio.gather(
            *(
                srv.post(
                    "/memory/update",
                    {"user_id": "u1", "id": record["id"], "content": f"edit {n}"},
                )
                for n in range(5)
            )
        )
        statuses = Counter(r.status_code for r in changes)
        assert statuses == Counter({200: 1, 409: 4}), [r.text[:200] for r in changes]
        assert all(
            r.json()["error"]["code"] == "VERSION_CONFLICT"
            for r in changes
            if r.status_code == 409
        )
        history = await srv.client.get(
            "/api/v1/memory/history", params={"user_id": "u1", "id": record["id"]}
        )
        assert history.status_code == 200, history.text
        states = [v["state"] for v in history.json()["data"]["versions"]]
        assert sorted(states) == ["current", "superseded"], states
        await _settled_and_clean(srv)


# ── reads during writes ─────────────────────────────────────────────────────


async def test_searches_during_writes_all_answer(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        resp, seed = await srv.say("s0", "u1", "S", 2)
        _ok([resp])
        await srv.settle()
        stop = asyncio.Event()
        answers: list[int] = []

        async def keep_searching() -> None:
            while not stop.is_set():
                answers.append((await srv.search("notes", user_id="u1")).status_code)

        async def keep_writing() -> list[str]:
            sent: list[str] = []
            for n in range(8):
                resp, got = await srv.say(f"s{n % 2 + 1}", "u1", f"W{n}", 2,
                                          at=(n + 1) * 60_000)  # fmt: skip
                _ok([resp])
                sent += got
            stop.set()
            return sent

        searching = [asyncio.create_task(keep_searching()) for _ in range(3)]
        sent = await keep_writing()
        await asyncio.gather(*searching)
        assert answers and set(answers) == {200}, Counter(answers)
        await _settled_and_clean(srv)
        assert_exactly_once(srv.episodes("u1"), [*seed, *sent])


async def test_profile_updates_racing_each_other_keep_every_fact(
    tmp_path: Path,
) -> None:
    """OME runs one profile update per episode; many at once must not lose one
    another's items (the profile document is read, changed and written)."""
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.delay["_ProfileUpdateReply"] = 0.2
        results = await asyncio.gather(
            *(
                srv.say(f"s{n}", "u1", f"P{n:02d}", 1, fact=True, at=n * 60_000)
                for n in range(8)
            )
        )
        _ok([r for r, _ in results])
        await _settled_and_clean(srv)
        profile = srv.profile_text("u1")
        missing = [m for _, s in results for m in s if f"Holds {m}" not in profile]
        assert missing == [], profile


async def test_a_forget_during_profile_updates(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.delay["_ProfileUpdateReply"] = 0.3
        resp, gone = await srv.say("s0", "u1", "G", 1, fact=True)
        _ok([resp])
        await srv.settle()
        writes = [
            srv.say(f"s{n}", "u1", f"K{n}", 1, fact=True, at=n * 60_000)
            for n in range(1, 5)
        ]
        results = await asyncio.gather(
            *writes, _forget(srv, "u1", {"kind": "session", "session_id": "s0"})
        )
        _ok([r[0] for r in results[:-1]] + [results[-1]])
        await _settled_and_clean(srv)
        profile = srv.profile_text("u1")
        kept = [m for r in results[:-1] for m in r[1]]
        assert all(f"Holds {m}" in profile for m in kept), profile
        assert f"Holds {gone[0]}" not in profile, profile


# ── the session lock ────────────────────────────────────────────────────────


async def test_the_session_lock_against_a_slow_model(tmp_path: Path) -> None:
    """A boundary call slower than ``session_lock_timeout_seconds``: the adds it
    holds up are refused (503), and sending those again loses and doubles
    nothing."""
    env = {"EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS": "1"}
    async with FaultServer.open(tmp_path / "root", env=env) as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.delay["_BatchReply"] = 2.5
        results = await asyncio.gather(
            *(srv.say("s1", "u1", f"L{n}", 2, at=n * 10_000) for n in range(3))
        )
        statuses = [r.status_code for r, _ in results]
        assert 503 in statuses, [r.text[:200] for r, _ in results]
        assert all(s in {200, 503} for s in statuses), statuses
        srv.upstream.delay.clear()
        for (resp, _sent), n in zip(results, range(3), strict=True):
            if resp.status_code != 200:
                again, _ = await srv.say("s1", "u1", f"L{n}", 2, at=n * 10_000)
                _ok([again])
        _ok([await srv.flush("s1")])
        await _settled_and_clean(srv)
        assert_exactly_once(srv.episodes("u1"), [m for _, s in results for m in s])
