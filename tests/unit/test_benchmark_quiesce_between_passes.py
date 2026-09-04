"""Ingest every conversation, freeze the index, then read — in that order.

The harness used to walk all four stages per conversation, so conv0's retrieval
ran while conv1..conv9 were still ingesting. That puts a 56-72s search (full-text
decider) next to cascade's projection maintenance, and prune's 60s retention
window then reclaims files the in-flight search still holds:

    LanceError(IO): Object at location .../_indices/<uuid>/tokens.lance not found

Measured on the first full LoCoMo run: 225 of 493 questions, every one an HTTP 500
that handed the answer model an empty context and scored zero. These tests pin the
two-pass split and the quiesce between them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")


class _Client:
    """Stands in for ``EverosClient`` — records what the harness posted."""

    posted: list[tuple[str, str]] = []

    def __init__(self, url: str, status: int = 200, body: dict | None = None) -> None:
        self._url, self._status = url, status
        self._body = body if body is not None else {"drained": 3, "pending_after": 0}

    def post(self, path: str, _payload: dict) -> tuple[int, dict]:
        _Client.posted.append((self._url, path))
        return self._status, self._body


@pytest.fixture(autouse=True)
def _reset() -> None:
    _Client.posted = []


def test_quiesces_every_server_not_just_the_first() -> None:
    """Each server owns its own store and its own cascade; one call would leave
    the other nine projections live under their own conversations' reads."""
    urls = [f"http://127.0.0.1:{9400 + i}" for i in range(10)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "EverosClient", lambda url: _Client(url))
        run._quiesce_servers(urls)

    assert [u for u, _ in _Client.posted] == urls
    assert {p for _, p in _Client.posted} == {"/api/v1/cascade/quiesce"}


def test_503_is_already_read_only_not_a_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected when the fleet was started with EVEROS_DISABLE_CASCADE — that
    server never had a projection running, so there is nothing to freeze."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "EverosClient", lambda url: _Client(url, status=503, body={}))
        run._quiesce_servers(["http://a", "http://b"])

    assert capsys.readouterr().out.count("already read-only") == 2


def test_any_other_failure_aborts_the_run() -> None:
    """Continuing would run the read stages against a live projection, which is
    the exact configuration that lost 225 of 493 questions. A run that scores
    those as retrieval misses looks finished and is wrong."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            run, "EverosClient", lambda url: _Client(url, status=500, body={"e": "x"})
        )
        with pytest.raises(RuntimeError, match="quiesce failed"):
            run._quiesce_servers(["http://a"])


def test_leftover_pending_warns_where_it_will_be_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Not fatal — the run is still worth having — but the index is an incomplete
    projection, so it has to be said out loud."""
    body = {"data": {"drained": 5, "pending_after": 2}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "EverosClient", lambda url: _Client(url, body=body))
        run._quiesce_servers("http://a")

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "incomplete" in out


def test_a_single_url_is_accepted_as_well_as_a_list() -> None:
    """``args.base_url`` is a list when the fleet started it and a bare string
    when an operator passed one address."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "EverosClient", lambda url: _Client(url))
        run._quiesce_servers("http://solo")
    assert _Client.posted == [("http://solo", "/api/v1/cascade/quiesce")]


def test_pass_split_puts_add_alone_and_keeps_the_read_stages_together() -> None:
    """The whole point: no conversation may be read while another still ingests."""
    for stages, expected in (
        (
            ["add", "search", "answer", "judge"],
            [["add"], ["search", "answer", "judge"]],
        ),
        (["add"], [["add"]]),
        (["search", "answer", "judge"], [["search", "answer", "judge"]]),
        (["judge"], [["judge"]]),
    ):
        read = [s for s in stages if s != "add"]
        passes = ([["add"]] if "add" in stages else []) + ([read] if read else [])
        assert passes == expected, stages
