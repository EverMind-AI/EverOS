"""DTO-layer path-safety validation for ``POST /api/v1/memory/add``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.entrypoints.api.routes.memorize import (
    MemorizeAddRequest,
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
        "../../../../etc",
        "..",
        ".",
        "a/b",
        "a/../b",
        "with space",
        "",
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
        "user@example.com",
        "user+tag",
        "user+tag@example.com",
    ],
)
def test_message_item_accepts_path_safe_sender_id(good_sender_id: str) -> None:
    assert _message(good_sender_id).sender_id == good_sender_id


def test_add_request_rejects_traversal_sender_id_in_messages() -> None:
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
