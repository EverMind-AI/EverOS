"""Model calls that fail: what a chat-memory server must do about it.

The real server (markdown, ``system.db``, LanceDB, cascade, OME, routes) over
a scripted upstream behind the real OpenAI provider (``tests.helpers.faults``):
each fault reaches the provider as the wire would carry it -- a timeout, a 5xx
burst, a 429, a reply that is not JSON, cut short or empty -- and goes through
the provider's own retries before the service sees it. For every fault:

* nothing a conversation said is lost or kept twice (the markers each message
  carries, counted over the live episodes);
* no staging file is left behind;
* markdown and the index agree once the cascade has run;
* a failure that is not absorbed is reported -- an error answer, a failed OME
  run, a ``degraded`` search -- never a silent success;
* the server keeps serving.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.helpers.faults import T0, FaultServer, assert_exactly_once

TRANSIENT = ["timeout", "500", "502", "503", "429", "malformed", "truncated", "cut"]
"""One failed attempt, then a good reply: the provider's retries absorb it."""

FAILED_RUN = {"failed", "dead_letter"}

NO_OME_RETRIES = """
[strategies.extract_user_profile]
max_retries = 0

[strategies.update_group_profile]
max_retries = 0
"""


async def _flushed(srv: FaultServer, session: str) -> None:
    resp = await srv.flush(session)
    assert resp.status_code == 200, resp.text
    await srv.settle()


async def _clean(srv: FaultServer) -> None:
    assert srv.staging() == []
    await srv.consistent()
    health = await srv.health()
    assert health["cascade"]["failed_permanent"] == 0, health["cascade"]


async def _group(srv: FaultServer, group: str) -> None:
    resp = await srv.client.put(
        f"/api/v1/groups/{group}",
        json={
            "asker_id": "u1",
            "roster": [
                {"op": "join", "member_id": m, "at": T0, "role": r}
                for m, r in (("u1", "owner"), ("u2", "member"), ("u3", "member"))
            ],
        },
    )
    assert resp.status_code == 200, resp.text


# ── the episode call ────────────────────────────────────────────────────────


@pytest.mark.parametrize("fault", [*TRANSIENT, "empty"])
async def test_one_failed_episode_call_is_absorbed(tmp_path: Path, fault: str) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.inject("EpisodeReply", fault)
        resp, sent = await srv.say("s1", "u1", "A")
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        assert srv.upstream.pending("EpisodeReply") == 0
        assert_exactly_once(srv.episodes("u1"), sent)
        assert await srv.searchable("u1") >= set(sent)
        await _clean(srv)


PERSISTENT = ["timeout", "503", "malformed", "null"]
"""Every attempt fails: the provider gives up and the service sees the error."""

ORPHANED_CELL = pytest.mark.xfail(
    strict=True,
    reason=(
        "extraction is not re-driven: a failed episode call leaves its memcell "
        "without an episode and the buffer consumed, so a retried flush finds "
        "nothing (docs/api.md, async queue 'Failure'; service/memory/"
        "_extraction.py). Extraction retry is being reworked in EverOS-fixingest."
    ),
)


@pytest.mark.parametrize("fault", PERSISTENT)
async def test_a_failed_episode_call_is_reported_and_a_resend_recovers(
    tmp_path: Path, fault: str
) -> None:
    """The documented recovery: a 503 means re-send the batch, then flush."""
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.inject("EpisodeReply", *[fault] * 10)
        resp, sent = await srv.say("s1", "u1", "A")
        assert resp.status_code == 200, resp.text
        failed = await srv.flush("s1")
        assert failed.status_code == 503, failed.text
        assert failed.json()["error"]["code"] == "EXTERNAL_SERVICE_UNAVAILABLE"
        srv.upstream._plan.clear()  # the model is back
        resp, _ = await srv.say("s1", "u1", "A")
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        assert_exactly_once(srv.episodes("u1"), sent)
        await _clean(srv)


@ORPHANED_CELL
async def test_a_failed_episode_call_is_extracted_by_the_retried_flush(
    tmp_path: Path,
) -> None:
    """A client that only retries the flush that failed loses the segment."""
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.down.add("EpisodeReply")
        resp, sent = await srv.say("s1", "u1", "A")
        assert resp.status_code == 200, resp.text
        assert (await srv.flush("s1")).status_code == 503
        srv.upstream.down.clear()
        await _flushed(srv, "s1")
        assert_exactly_once(srv.episodes("u1"), sent)


async def test_a_failed_group_episode_call_is_taken_in_full_on_resend(
    tmp_path: Path,
) -> None:
    """A group skips messages it holds -- but not those of a failed batch."""
    async with FaultServer.open(tmp_path / "root") as srv:
        await _group(srv, "trip")
        srv.upstream.cut_on_add = True
        srv.upstream.down.add("EpisodeReply")
        resp, sent = await srv.say("s1", "u1", "G", group="trip")
        assert resp.status_code == 503, resp.text
        srv.upstream.down.clear()
        resp, _ = await srv.say("s1", "u1", "G", group="trip")
        assert resp.status_code == 200, resp.text
        assert not resp.json()["data"].get("duplicates"), resp.text
        await srv.settle()
        assert_exactly_once(srv.episodes("trip"), sent)
        await _clean(srv)


# ── the boundary call ───────────────────────────────────────────────────────


@pytest.mark.parametrize("fault", ["timeout", "503", "429", "malformed"])
async def test_one_failed_boundary_call_is_absorbed(tmp_path: Path, fault: str) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.inject("_BatchReply", fault)
        resp, sent = await srv.say("s1", "u1", "B")
        assert resp.status_code == 200, resp.text
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await _clean(srv)


async def test_a_boundary_reply_cut_short_fails_the_add_at_once(tmp_path: Path) -> None:
    """The boundary call does not resend a reply cut at ``length``: the add is
    refused whole (503) and the batch can be sent again."""
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.inject("_BatchReply", "truncated")
        resp, sent = await srv.say("s1", "u1", "B")
        assert resp.status_code == 503, resp.text
        assert srv.upstream.count("_BatchReply") == 1
        resp, _ = await srv.say("s1", "u1", "B")
        assert resp.status_code == 200, resp.text
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await _clean(srv)


async def test_a_boundary_call_that_keeps_failing_stores_nothing_and_a_resend_recovers(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.down.add("_BatchReply")
        resp, sent = await srv.say("s1", "u1", "B")
        assert resp.status_code == 503, resp.text
        assert (await srv.flush("s1")).json()["data"]["status"] == "no_extraction"
        srv.upstream.down.clear()
        resp, _ = await srv.say("s1", "u1", "B")
        assert resp.status_code == 200, resp.text
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await _clean(srv)


# ── the picture call ────────────────────────────────────────────────────────


def _picture_line(at: int, marker: str) -> dict[str, object]:
    from tests.helpers.faults import line
    from tests.helpers.groups import b64, png

    return line(
        "u1",
        [
            {"type": "text", "text": f"look {marker}"},
            {"type": "image", "base64": b64(png(at % 250)), "ext": "png"},
        ],
        at,
    )


@pytest.mark.parametrize("fault", ["503", "malformed", "truncated"])
async def test_a_failed_picture_call(tmp_path: Path, fault: str) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.inject("ImageEpisodeReply", fault)
        resp = await srv.add("s1", _picture_line(0, "ZQP00"), _picture_line(1, "ZQP01"))
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        assert_exactly_once(srv.episodes("u1"), ["ZQP00", "ZQP01"])
        await _clean(srv)


async def test_a_picture_call_that_keeps_failing_and_a_resend(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.down.add("ImageEpisodeReply")
        resp = await srv.add("s1", _picture_line(0, "ZQP00"))
        assert resp.status_code == 200, resp.text
        failed = await srv.flush("s1")
        assert failed.status_code == 503, failed.text
        srv.upstream.down.clear()
        resp = await srv.add("s1", _picture_line(0, "ZQP00"))
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        assert_exactly_once(srv.episodes("u1"), ["ZQP00"])
        await _clean(srv)


# ── the profile and group-profile calls (OME) ───────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a profile update that failed is dead-lettered and never re-driven; the "
        "next update reads only its own segment, so the failed window's facts "
        "never reach the profile (memory/strategies/extract_user_profile.py). "
        "Profile cadence is being reworked in EverOS-fixingest."
    ),
)
async def test_a_failed_profile_update_is_reported_and_caught_up(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root", ome_toml=NO_OME_RETRIES) as srv:
        srv.upstream.down.add("_ProfileUpdateReply")
        resp, first = await srv.say("s1", "u1", "P", 2, fact=True)
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        runs = await srv.runs("extract_user_profile")
        assert runs and runs[-1][0] in FAILED_RUN and runs[-1][1], runs
        srv.upstream.down.clear()
        resp, second = await srv.say("s2", "u1", "Q", 2, fact=True, at=60_000)
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s2")
        profile = srv.profile_text("u1")
        missing = [m for m in [*first, *second] if f"Holds {m}" not in profile]
        assert missing == [], profile
        await _clean(srv)


@pytest.mark.parametrize("fault", ["timeout", "503", "malformed"])
async def test_one_failed_profile_call_is_absorbed(tmp_path: Path, fault: str) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.upstream.inject("_ProfileUpdateReply", fault)
        resp, facts = await srv.say("s1", "u1", "P", 2, fact=True)
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        profile = srv.profile_text("u1")
        assert all(f"Holds {m}" in profile for m in facts), profile
        await _clean(srv)


async def test_a_failed_group_profile_update_is_kept_and_caught_up(
    tmp_path: Path,
) -> None:
    """A group profile update that fails keeps its batch and backs off
    (``group_profile_idle_seconds``, here 2 s); the next update reads it."""
    env = {"EVEROS_GROUPS__GROUP_PROFILE_IDLE_SECONDS": "2"}
    async with FaultServer.open(tmp_path / "root", env=env) as srv:
        await _group(srv, "trip")
        srv.upstream.down.add("GroupProfileReply")
        resp, first = await srv.say("s1", "u1", "G", 2, group="trip")
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        assert srv.upstream.count("GroupProfileReply") >= 1
        srv.upstream.down.clear()
        await asyncio.sleep(2.5)  # past the back-off
        resp, second = await srv.say("s1", "u2", "H", 2, group="trip", at=60_000)
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        text = "\n".join(
            path.read_text("utf-8")
            for path in srv.root.rglob("*.md")
            if "group_profile" in path.as_posix() or "/profile/" in path.as_posix()
        )
        missing = [m for m in [*first, *second] if m not in text]
        assert missing == [], text
        await _clean(srv)


# ── the decider ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("faults", "outcome"),
    [
        (["timeout"] * 4, "fallback"),
        (["503"] * 4, "fallback"),
        (["malformed"] * 4, "fallback"),
        (["empty"] * 4, "fallback"),
        (["truncated"] * 4, "salvaged"),
        (["503", "malformed"], None),
    ],
)
async def test_decider_faults_degrade_the_search_and_say_so(
    tmp_path: Path, faults: list[str], outcome: str | None
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        resp, sent = await srv.say("s1", "u1", "D")
        assert resp.status_code == 200, resp.text
        await _flushed(srv, "s1")
        srv.upstream.inject("_DecisionReply", *faults)
        resp = await srv.search("notes", user_id="u1")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        degraded = [d.get("reason") or d for d in data["degraded"]]
        if outcome is None:
            assert degraded == [], data["degraded"]
        else:
            assert data["degraded"], data
            assert outcome in str(data["degraded"]), data["degraded"]
        assert set(sent) <= set(__import__("re").findall(r"ZQ[A-Z0-9]+", resp.text))
        health = await srv.health()
        assert (health["decider"]["failed_rounds"] > 0) == (outcome is not None)
        again = await srv.search("notes", user_id="u1")
        assert again.status_code == 200 and again.json()["data"]["degraded"] == []
