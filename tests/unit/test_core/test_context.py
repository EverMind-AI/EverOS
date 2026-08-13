"""``core.context`` — request-scoped contextvar propagation.

The request id lives in a ``ContextVar`` so it flows across ``await``
boundaries (HTTP middleware → service → infra → logs) without being
threaded through every call signature.
"""

from __future__ import annotations

from everos.core.context import (
    get_request_id,
    reset_request_id,
    resolve_request_id,
    set_request_id,
)


def test_get_request_id_defaults_to_none() -> None:
    assert get_request_id() is None


def test_set_request_id_roundtrip() -> None:
    token = set_request_id("abc123")
    try:
        assert get_request_id() == "abc123"
    finally:
        reset_request_id(token)


def test_reset_request_id_restores_previous() -> None:
    token = set_request_id("first")
    assert get_request_id() == "first"
    reset_request_id(token)
    assert get_request_id() is None


def test_resolve_returns_bound_id_when_present() -> None:
    token = set_request_id("deadbeef" * 4)
    try:
        assert resolve_request_id() == "deadbeef" * 4
    finally:
        reset_request_id(token)


def test_resolve_mints_32hex_when_absent() -> None:
    rid = resolve_request_id()
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)
