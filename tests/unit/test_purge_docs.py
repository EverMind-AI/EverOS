"""The purge docs carry what spec §5.11 asks of them (A26).

Each document must name the facts a reader needs (keywords below), and the
curl example of ``docs/api.md`` must run as written once ``$PORT``,
``<NONCE>`` and ``<JOB_ID>`` are filled in -- no other placeholder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"

KEYWORDS = {
    "api.md": [
        "/api/v2/memory/purge",
        "/purge/confirm",
        "/purge/{job_id}",
        "irreversible",
        "entire memory root",
        "PURGE_NONCE_INVALID",
        "purge_in_progress",
        "does not block",
        "PURGE_IN_PROGRESS",
        "A refused write wrote nothing",
        "restart the server, then purge again",
        "only once",
        "If the `202` is lost",
        '"default"',
        "summary of all `0`",
        "Known gaps",
        "There is no authentication",
        "reverse proxy",
        "running** job",
        "lower case",
        "fresh temporary",
    ],
    "storage_layout.md": [
        "Online clearing",
        "Entry ids keep counting",
        "OME run history stays",
        "Older LanceDB table versions",
        "collection_prefix",
    ],
    "configuration.md": [
        "### `[purge]`",
        "run_budget_seconds",
        "nonce_ttl_seconds",
        "count_timeout_seconds",
        "Give every memory root its own prefix",
    ],
    "cli.md": [
        "CLI commands run as usual",
        "resumes",
        "Do not run `everos cascade backfill`",
    ],
}


@pytest.mark.parametrize(
    ("doc", "keyword"), [(doc, kw) for doc, kws in KEYWORDS.items() for kw in kws]
)
def test_doc_names_it(doc: str, keyword: str) -> None:
    assert keyword in (DOCS / doc).read_text(encoding="utf-8")


def _purge_section() -> str:
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    start = text.index("### POST /api/v2/memory/purge\n")
    return text[start : text.index("\n### ", start + 1)]


def test_curl_example_has_only_the_three_placeholders() -> None:
    blocks = re.findall(r"```bash\n(.*?)```", _purge_section(), re.S)
    assert len(blocks) == 1
    found = set(re.findall(r"<[^<>\s]+>", blocks[0]))
    assert found == {"<NONCE>", "<JOB_ID>"}
    assert "$PORT" in blocks[0]


def test_purge_docs_drop_the_removed_settle_wait() -> None:
    """``settle_wait_seconds`` is gone (spec v3 D19)."""
    assert "settle_wait_seconds" not in (DOCS / "configuration.md").read_text(
        encoding="utf-8"
    )
