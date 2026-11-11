"""T26.1, T26.2: the documented commands run as written; the spec matches."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_purge_docs import KEYWORDS

from .journey import Evidence, evidence_base

pytestmark = pytest.mark.acceptance

DOCS = Path(__file__).parents[3] / "docs"


def _commands() -> list[str]:
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    start = text.index("### POST /api/v2/memory/purge\n")
    section = text[start : text.index("\n### ", start + 1)]
    (block,) = re.findall(r"```bash\n(.*?)```", section, re.S)
    joined = re.sub(r"\\\n\s*", " ", block)
    return [line.strip() for line in joined.splitlines() if line.startswith("curl")]


def _run(command: str) -> dict[str, Any]:
    out = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def test_t26_1_the_api_doc_curl_example_runs_as_written(
    make_purge_server: Any, tmp_path: Path
) -> None:
    srv = make_purge_server()
    ev = Evidence("T26.1", evidence_base(tmp_path), [srv])
    try:
        request, confirm, poll = _commands()
        port = str(srv.port)
        fill = {"$PORT": port}
        first = _run(request.replace("$PORT", port))
        fill["<NONCE>"] = first["data"]["nonce"]
        started = _run(
            confirm.replace("$PORT", port).replace("<NONCE>", fill["<NONCE>"])
        )
        job_id = started["data"]["job_id"]
        get = poll.replace("$PORT", port).replace("<JOB_ID>", job_id)
        deadline = time.monotonic() + 60
        status = ""
        while time.monotonic() < deadline:
            status = _run(get)["data"]["status"]
            if status in ("succeeded", "failed"):
                break
            time.sleep(0.5)
        again = _run(request.replace("$PORT", port))
        ev.json(
            "run",
            {
                "commands": [request, confirm, poll],
                "first": first,
                "confirm": started,
                "status": status,
                "again": again,
            },
        )
        assert status == "succeeded"
        summary = again["data"]["summary"]
        assert (summary["markdown_total"], summary["sqlite_total"]) == (0, 0)
        for doc, words in KEYWORDS.items():
            text = (DOCS / doc).read_text(encoding="utf-8")
            assert all(w in text for w in words), doc
    finally:
        ev.close()


def test_t26_2_served_spec_matches_docs(make_purge_server: Any) -> None:
    srv = make_purge_server(ENV="DEV")
    served = srv.call("GET", "/openapi.json", version="").json()
    committed = json.loads((DOCS / "openapi.json").read_text(encoding="utf-8"))
    purge_paths = [p for p in served["paths"] if "/memory/purge" in p]
    assert sorted(purge_paths) == sorted(
        f"/api/{v}/memory/purge{s}"
        for v in ("v1", "v2")
        for s in ("", "/confirm", "/{job_id}")
    )
    assert served == committed
