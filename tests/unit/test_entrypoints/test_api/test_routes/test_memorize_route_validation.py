"""DTO-layer path-safety validation for ``POST /api/v1/memory/add``.

``sender_id`` flows through to ``owner_id`` and is joined into the episode
write path as a directory segment, so it must carry the same path-traversal
guard as ``app_id`` / ``project_id`` (charset whitelist + ``.``/``..``
rejection). These tests pin that guard at the DTO layer; the writer-level
containment backstop is covered in
``tests/unit/test_core/test_persistence/test_markdown/test_writer.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.entrypoints.api.routes.memorize import (
    MemorizeAddRequest,
    MemorizeFlushRequest,
    MessageItemDTO,
)


def _message(sender_id: str) -> MessageItemDTO:
    return MessageItemDTO(
        sender_id=sender_id,
        role="user",
        timestamp=1_700_000_000_000,
        content="x",
    )


@pytest.mark.parametrize(
    "bad_sender_id",
    [
        "../../../../etc",  # classic traversal
        "..",  # reserved parent token
        ".",  # reserved current-dir token
        "a/b",  # embedded path separator
        "a/../b",  # separator + traversal mid-string
        "with space",  # outside the charset whitelist
        "",  # empty (min_length)
    ],
)
def test_message_item_rejects_unsafe_sender_id(bad_sender_id: str) -> None:
    with pytest.raises(ValidationError):
        _message(bad_sender_id)


@pytest.mark.parametrize(
    "good_sender_id",
    [
        "u1",
        "u_jason",
        "user-123",
        "a.b_c-1",
        "default",
        "user@example.com",  # email-style id (``@`` + dotted domain)
        "user+tag",  # plus-addressing
        "user+tag@example.com",  # both, combined
        "User-A",  # scope restrictions do not change sender identities
    ],
)
def test_message_item_accepts_path_safe_sender_id(good_sender_id: str) -> None:
    assert _message(good_sender_id).sender_id == good_sender_id


def test_add_request_rejects_traversal_sender_id_in_messages() -> None:
    # The guard fires through the nested message list, not just on a bare DTO.
    with pytest.raises(ValidationError):
        MemorizeAddRequest(
            session_id="s1",
            app_id="default",
            project_id="default",
            messages=[
                {
                    "sender_id": "../../../../ESCAPED",
                    "role": "user",
                    "timestamp": 1_700_000_000_000,
                    "content": "secret",
                }
            ],
        )


@pytest.mark.parametrize("sender_id", ["default_app", "default_project"])
def test_reserved_scope_aliases_remain_valid_sender_ids(sender_id: str) -> None:
    """Scope aliases are reserved by field, not globally across identities."""
    assert _message(sender_id).sender_id == sender_id


@pytest.mark.parametrize(
    "request_type, field, value",
    [
        (MemorizeAddRequest, "app_id", "default_app"),
        (MemorizeAddRequest, "project_id", "default_project"),
        (MemorizeFlushRequest, "app_id", "default_app"),
        (MemorizeFlushRequest, "project_id", "default_project"),
    ],
)
def test_memory_requests_reject_reserved_scope_aliases(
    request_type: type[MemorizeAddRequest] | type[MemorizeFlushRequest],
    field: str,
    value: str,
) -> None:
    payload = {
        "session_id": "s1",
        "app_id": "default",
        "project_id": "default",
    }
    payload[field] = value
    if request_type is MemorizeAddRequest:
        payload["messages"] = [_message("u1")]
    with pytest.raises(ValidationError):
        request_type.model_validate(payload)


@pytest.mark.parametrize(
    "request_type, field, value",
    [
        (MemorizeAddRequest, "project_id", "Project-A"),
        (MemorizeFlushRequest, "project_id", "Project-A"),
        (MemorizeAddRequest, "app_id", "DEFAULT_APP"),
        (MemorizeFlushRequest, "app_id", "DEFAULT_APP"),
        (MemorizeAddRequest, "app_id", ".index"),
        (MemorizeFlushRequest, "app_id", ".tmp"),
        (MemorizeAddRequest, "app_id", "x" * 129),
        (MemorizeFlushRequest, "project_id", "safe\n"),
    ],
)
def test_memory_requests_reject_nonportable_scopes(
    request_type: type[MemorizeAddRequest] | type[MemorizeFlushRequest],
    field: str,
    value: str,
) -> None:
    payload = {
        "session_id": "s1",
        "app_id": "default",
        "project_id": "default",
    }
    payload[field] = value
    if request_type is MemorizeAddRequest:
        payload["messages"] = [_message("User-A")]

    with pytest.raises(ValidationError):
        request_type.model_validate(payload)
