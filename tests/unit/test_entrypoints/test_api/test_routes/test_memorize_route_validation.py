"""DTO-layer path-safety validation for ``POST /api/v1/memory/add``.

``sender_id`` flows through to ``owner_id`` and is joined into the episode
write path as a directory segment, so it must carry the same path-traversal
guard as ``app_id`` / ``project_id`` (charset whitelist + ``.``/``..``
rejection). These tests pin that guard at the DTO layer; the writer-level
containment backstop is covered in
``tests/unit/test_core/test_persistence/test_markdown/test_writer.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.routes.memorize import (
    MemorizeAddRequest,
    MessageItemDTO,
)


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    """FastAPI app with no lifespan; nothing past DTO validation is reached."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    app = create_app(lifespan_providers=[])
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    load_settings.cache_clear()


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


def test_message_item_rejects_tool_role_without_call_id() -> None:
    # An orphan tool row used to travel to ``_boundary`` and 500 from
    # inside extraction; it is refused at the DTO now.
    with pytest.raises(ValidationError, match="tool_call_id"):
        MessageItemDTO(
            sender_id="agent",
            role="tool",
            timestamp=1_700_000_000_000,
            content="x",
        )


def test_message_item_accepts_tool_role_with_call_id() -> None:
    m = MessageItemDTO(
        sender_id="agent",
        role="tool",
        timestamp=1_700_000_000_000,
        content="x",
        tool_call_id="call_1",
    )
    assert m.tool_call_id == "call_1"


async def test_add_orphan_tool_row_is_422_not_500(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/memory/add",
        json={
            "session_id": "s1",
            "messages": [
                {
                    "sender_id": "agent",
                    "role": "tool",
                    "timestamp": 1_700_000_000_000,
                    "content": "x",
                }
            ],
        },
    )
    assert resp.status_code == 422
    assert "tool_call_id" in resp.text


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


def test_add_request_can_defer_extraction() -> None:
    request = MemorizeAddRequest(
        session_id="s1",
        messages=[_message("u1")],
        defer_extraction=True,
    )

    assert request.defer_extraction is True
