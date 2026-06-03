"""E2E: multimodal /add parses HTML (base64) and http(s) uri end-to-end.

Scope: full HTTP stack (``create_app()`` + ``AsyncClient``) → ingest →
multimodal parse → unprocessed_buffer. Proves the three paths the unit
tests can only mock:

1. ``type="html"`` + base64 + ``ext="html"`` — the normal HTML-file call.
2. ``type="html"`` + ``https`` uri — everalgo fetches the page and
   dispatches by the response Content-Type.
3. ``type="html"`` + ``file://`` uri — EverOS reads the file locally and
   hands everalgo hydrated bytes (the library never touches the fs).

Real multimodal LLM (creds via ``.env``) + real public internet, so the
module is marked ``live_llm``. Skipped when the ``[multimodal]`` extra is
absent.

White-box surface: reads the ``text`` column of ``unprocessed_buffer``
(the derived text the ingest stage produced from the parsed content) to
assert the parsed payload actually flowed into the buffer.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text as sql_text

pytest.importorskip("everalgo.parser")

pytestmark = pytest.mark.live_llm


async def _buffer_text(session_id: str) -> str:
    """Concatenated derived ``text`` of all buffer rows for a session."""
    from everos.infra.persistence.sqlite import get_engine

    async with get_engine().connect() as conn:
        rows = (
            await conn.execute(
                sql_text("SELECT text FROM unprocessed_buffer WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).all()
    return "\n".join(str(r[0]) for r in rows)


async def test_add_html_base64_parsed_into_buffer(
    async_client: httpx.AsyncClient,
) -> None:
    """A base64 HTML file is parsed and its text lands in the buffer."""
    html = (
        b"<html><body><h1>Release</h1>"
        b"<p>Version 9.9.9 ships Dark Mode.</p></body></html>"
    )
    sid = "e2e-mm-html-b64"
    resp = await async_client.post(
        "/api/v1/memory/add",
        json={
            "session_id": sid,
            "messages": [
                {
                    "sender_id": "alice",
                    "role": "user",
                    "timestamp": 1780304400000,
                    "content": [
                        {
                            "type": "html",
                            "base64": base64.b64encode(html).decode(),
                            "ext": "html",
                            "name": "notes.html",
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    buffered = await _buffer_text(sid)
    assert "9.9.9" in buffered


async def test_add_html_https_uri_parsed_into_buffer(
    async_client: httpx.AsyncClient,
) -> None:
    """An https uri is fetched + parsed and its text lands in the buffer."""
    sid = "e2e-mm-html-uri"
    resp = await async_client.post(
        "/api/v1/memory/add",
        json={
            "session_id": sid,
            "messages": [
                {
                    "sender_id": "alice",
                    "role": "user",
                    "timestamp": 1780304400000,
                    "content": [{"type": "html", "uri": "https://example.com"}],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    buffered = await _buffer_text(sid)
    assert "example domain" in buffered.lower()


async def test_add_html_file_uri_parsed_into_buffer(
    async_client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    """A file:// html asset is read locally (hydrated) + parsed into buffer.

    Exercises EverOS-side file:// support: the parser receives bytes, never
    the path. Default allowlist is empty (local-first) so the temp file reads.
    """
    doc = tmp_path / "release.html"
    doc.write_text("<html><body><p>Version 9.9.9 ships Dark Mode.</p></body></html>")
    sid = "e2e-mm-html-file"
    resp = await async_client.post(
        "/api/v1/memory/add",
        json={
            "session_id": sid,
            "messages": [
                {
                    "sender_id": "alice",
                    "role": "user",
                    "timestamp": 1780304400000,
                    "content": [{"type": "html", "uri": f"file://{doc}"}],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    buffered = await _buffer_text(sid)
    assert "9.9.9" in buffered
