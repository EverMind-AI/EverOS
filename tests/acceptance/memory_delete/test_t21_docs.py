"""T21.1 <- A21 and TB13.1 <- B13: the documentation is a deliverable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.acceptance

_DOCS = Path(__file__).resolve().parents[3] / "docs"


def test_t21_1_and_tb13_1_docs_say_what_the_server_does(make_server: Any) -> None:
    server = make_server()
    server.start(ENV="dev")  # /openapi.json is served only in dev
    api = (_DOCS / "api.md").read_text(encoding="utf-8")
    for needle in (
        "### POST /api/v2/memory/delete\n",
        "### POST /api/v2/memory/delete_by_ids\n",
        "#### What each scope deletes",
        "`SESSION_BUSY`",
        "nothing was deleted, retry later",
        "deletion incomplete; retry the same request",
        "One window",  # the index's short visible window
        "everos cascade rebuild",
        "Known gap",
        "Deleting one source of a merged episode keeps the merged episode",
        "never records `note`",
        "`null` for an episode Reflection merged",
    ):
        assert needle in api, needle
    assert "not available yet" not in (_DOCS / "mcp.md").read_text(encoding="utf-8")
    config = (_DOCS / "configuration.md").read_text(encoding="utf-8")
    assert "`delete_wait_seconds`" in config and "`delete_fence_warn_entries`" in config

    served = httpx.get(f"{server.base}/openapi.json", timeout=10).json()
    committed = json.loads((_DOCS / "openapi.json").read_text(encoding="utf-8"))
    assert served == committed
    assert "/api/v2/memory/delete" in served["paths"]
    assert "/api/v2/memory/delete_by_ids" in served["paths"]
