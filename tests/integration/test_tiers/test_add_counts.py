"""``/add`` returns what ingest dropped, through the real assembly.

The ingest counts (pictures refused, items the parser could not read) used to
stop at the ingest layer: ``/add`` answered ``message_count`` and ``status``
only. Here a batch carries one picture over the per-message limit and one
item that is not a picture at all; both are refused, the text is kept, and
the response says so.
"""

from __future__ import annotations

import base64
import io

from httpx import AsyncClient
from PIL import Image


def _png(colour: tuple[int, int, int]) -> dict[str, str]:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(buffer, format="PNG")
    return {
        "type": "image",
        "base64": base64.b64encode(buffer.getvalue()).decode(),
        "ext": "png",
    }


async def test_add_returns_the_refused_pictures(tier1_runtime: AsyncClient) -> None:
    five = [_png((10 * i, 20, 30)) for i in range(5)]  # [images] max_per_message 4
    not_a_picture = {
        "type": "image",
        "base64": base64.b64encode(b"plain text, not a picture").decode(),
        "ext": "png",
    }
    resp = await tier1_runtime.post(
        "/api/v2/memory/add",
        json={
            "session_id": "s_counts",
            "messages": [
                {
                    "sender_id": "u_alice",
                    "role": "user",
                    "timestamp": 1740564000000,
                    "content": [{"type": "text", "text": "my holiday"}, *five],
                },
                {
                    "sender_id": "u_alice",
                    "role": "user",
                    "timestamp": 1740564060000,
                    "content": [{"type": "text", "text": "and this"}, not_a_picture],
                },
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["message_count"] == 2
    assert (data["rejected"], data["failed"]) == (2, 0)


async def test_a_clean_batch_reports_zero(tier1_runtime: AsyncClient) -> None:
    resp = await tier1_runtime.post(
        "/api/v2/memory/add",
        json={
            "session_id": "s_clean",
            "messages": [
                {
                    "sender_id": "u_alice",
                    "role": "user",
                    "timestamp": 1740564000000,
                    "content": "hello",
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert (data["rejected"], data["failed"]) == (0, 0)
