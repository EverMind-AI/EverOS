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
