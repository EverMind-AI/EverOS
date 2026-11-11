"""The harness itself: the fake upstream answers every call, the seed fills
every table in every scope (acceptance P0 -- a table empty before a purge
reads 0 after it whether or not anything was deleted)."""

from __future__ import annotations

import json

import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.fakes.llm import (
    AGENT_TRACK_OPERATORS,
    KNOWLEDGE_OPERATORS,
    USER_TRACK_OPERATORS,
)

from .observe import scope_counts, wait_idle
from .seed import WORLD, scope_token, seed_purge_world

pytestmark = pytest.mark.acceptance

KNOWN = {
    *USER_TRACK_OPERATORS.values(),
    *AGENT_TRACK_OPERATORS.values(),
    *KNOWLEDGE_OPERATORS.values(),
}


def test_fake_upstream_sees_schema_names(make_purge_server, fake) -> None:
    srv = make_purge_server()
    token = scope_token("a1", "p1")
    where = {"app_id": "a1", "project_id": "p1"}
    assert (
        srv.post(
            "/memory/add", {"session_id": "s1", "messages": chat("u1", token), **where}
        ).status_code
        == 200
    )
    assert srv.post("/memory/flush", {"session_id": "s1", **where}).status_code == 200
    wait_idle(srv.base, srv.root)
    first = fake.first_llm_body
    assert first is not None, "the server never called the LLM"
    assert first["response_format"]["json_schema"]["name"] in KNOWN, first
    assert fake.unknown_names == [], fake.unknown_names
    assert set(fake.names) <= KNOWN


def test_seed_reaches_every_table(make_purge_server, fake) -> None:
    srv = make_purge_server()
    log = seed_purge_world(srv, fake, WORLD)
    counts = {scope: scope_counts(srv.root, *scope) for scope in WORLD}
    print(json.dumps({f"{a}/{p}": c for (a, p), c in counts.items()}, indent=1))
    assert fake.unknown_names == [], fake.unknown_names
    empty = {
        f"{a}/{p}": sorted(k for k, n in c.items() if n == 0)
        for (a, p), c in counts.items()
    }
    assert all(not v for v in empty.values()), empty
    assert set(log.scopes) == set(WORLD)
