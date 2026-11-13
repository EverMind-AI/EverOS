"""Seed every table a purge clears, through the server's own HTTP API (H5).

Per scope: two users each talk to agent ``ag1`` in two sessions (memcell,
conversation status, episodes; atomic facts, foresights, the agent case and
its skill, clusters and the profile through OME), one deferred add (the
unprocessed buffer), one knowledge document, and one async add that succeeds
plus one the fake upstream fails (``memorize_inbox``). Reflection runs once
over everything at the end; ``md_change_state`` is the cascade's own.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from tests.acceptance.memory_delete.conftest import chat
from tests.fakes.llm import UNUSABLE_REPLY

from .conftest import PurgeServer
from .fake_upstream import FakeUpstream, Override
from .observe import wait_idle

USERS = ("u1", "u2")
AGENT = "ag1"


@dataclass
class ScopeSeed:
    token: str  # only this scope's requests carry it (proxy rules match it)
    users: list[str]
    agent: str
    sessions: list[str]
    doc_id: str
    inbox_request_ids: list[str] = field(default_factory=list)


@dataclass
class SeedLog:
    scopes: dict[tuple[str, str], ScopeSeed] = field(default_factory=dict)


def scope_token(app: str, project: str) -> str:
    return "zq" + hashlib.sha256(f"{app}/{project}".encode()).hexdigest()[:6]


def _ok(resp: httpx.Response, *codes: int) -> dict[str, Any]:
    assert resp.status_code in (codes or (200,)), resp.text
    return resp.json()


def agent_chat(user: str, token: str) -> list[dict[str, Any]]:
    """A user-agent exchange with three tool calls (an agent case needs at
    least three tool rounds)."""
    msgs = chat(user, token, assistant=AGENT, n=2)
    at = msgs[-1]["timestamp"]
    for i in range(3):
        call = {
            "id": f"call_{token}_{i}",
            "type": "function",
            "function": {
                "name": "search_museums",
                "arguments": f'{{"q": "{token}", "page": {i}}}',
            },
        }
        msgs += [
            {
                "sender_id": AGENT,
                "role": "assistant",
                "timestamp": at + 1000,
                "content": f"Let me look up the {token} museum, page {i}.",
                "tool_calls": [call],
            },
            {
                "sender_id": AGENT,
                "role": "tool",
                "timestamp": at + 2000,
                "content": f"The {token} museum, page {i}: opens at 9.",
                "tool_call_id": call["id"],
            },
        ]
        at += 2000
    msgs.append(
        {
            "sender_id": AGENT,
            "role": "assistant",
            "timestamp": at + 1000,
            "content": f"The {token} museum opens at 9 in the morning.",
        }
    )
    return msgs


def talk(
    srv: PurgeServer, user: str, session: str, token: str, where: dict[str, str]
) -> str:
    """One user-agent session, added and flushed."""
    _ok(
        srv.post(
            "/memory/add",
            {
                "session_id": session,
                "messages": agent_chat(user, token),
                **where,
            },
        )
    )
    _ok(srv.post("/memory/flush", {"session_id": session, **where}))
    return session


def seed_scope(
    srv: PurgeServer, fake: FakeUpstream, app: str, project: str
) -> ScopeSeed:
    token = scope_token(app, project)
    where = {"app_id": app, "project_id": project}
    sessions = []
    for user in USERS:
        for i in (1, 2):
            sessions.append(talk(srv, user, f"{user}-s{i}", token, where))
    _ok(
        srv.post(
            "/memory/add",
            {
                "session_id": "buffered",
                "messages": chat("u1", token),
                "defer_extraction": True,
                **where,
            },
        )
    )
    doc = _ok(
        srv.call(
            "POST",
            "/knowledge/documents",
            data={"title": f"Hiking {token}", **where},
            files={
                "file": (
                    "guide.txt",
                    f"Mountain hiking safety guide {token}.".encode(),
                    "text/plain",
                )
            },
        ),
        201,
    )["data"]
    bad = f"{token}bad"
    fake.override(Override(pattern=bad, reply=UNUSABLE_REPLY))
    request_ids = []
    for session, words in (("async-ok", token), ("async-bad", bad)):
        accepted = _ok(
            srv.post(
                "/memory/add",
                {
                    "session_id": session,
                    "messages": chat("u2", words),
                    "async_mode": True,
                    **where,
                },
            ),
            202,
        )
        request_ids.append(accepted["request_id"])
    return ScopeSeed(
        token=token,
        users=list(USERS),
        agent=AGENT,
        sessions=sessions,
        doc_id=doc["doc_id"],
        inbox_request_ids=request_ids,
    )


def seed_purge_world(
    srv: PurgeServer, fake: FakeUpstream, scopes: Sequence[tuple[str, str]]
) -> SeedLog:
    log = SeedLog()
    for app, project in scopes:
        log.scopes[(app, project)] = seed_scope(srv, fake, app, project)
    wait_idle(srv.base, srv.root, timeout=300)
    # Second round: profile extraction reads the owner's episodes from the
    # index, and with an instant model it runs before the cascade indexed the
    # first round; by now it has, so this round's profile finds them.
    for (app, project), seeded in log.scopes.items():
        where = {"app_id": app, "project_id": project}
        for user in USERS:
            seeded.sessions.append(talk(srv, user, f"{user}-s3", seeded.token, where))
    wait_idle(srv.base, srv.root, timeout=300)
    trig = _ok(
        srv.post(
            "/ome/trigger", {"name": "reflect_episodes", "force": True, "timeout": 120}
        )
    )
    assert trig["status"] == "ok", trig  # not enveloped (TriggerResponse)
    wait_idle(srv.base, srv.root, timeout=300)
    return log


WORLD = (("a1", "p1"), ("a1", "p2"), ("a2", "p1"), ("default", "default"))
"""S, its sibling project, another app, and the default scope (H5)."""
