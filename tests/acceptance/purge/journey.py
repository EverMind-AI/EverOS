"""What every purge case does: seed the world, check P0, purge, keep evidence."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .conftest import PurgeServer
from .fake_upstream import FakeUpstream
from .observe import rows, scope_counts, system_db
from .seed import WORLD, SeedLog

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


_STARTED: set[Path] = set()


@dataclass
class Evidence:
    """``$PURGE_EVIDENCE_DIR/<case>/`` (default: under the test's tmp dir)."""

    case: str
    base: Path
    servers: list[PurgeServer] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Start the case afresh: an earlier run's requests and logs would
        # read as this run's. Mutation records are kept (written separately).
        path = self.base / self.case
        if path in _STARTED:
            return
        _STARTED.add(path)
        if path.is_dir():
            for old in path.iterdir():
                if old.is_file() and not old.name.startswith("mutation"):
                    old.unlink()

    @property
    def dir(self) -> Path:
        path = self.base / self.case
        path.mkdir(parents=True, exist_ok=True)
        return path

    def json(self, name: str, obj: Any) -> None:
        (self.dir / f"{name}.json").write_text(
            json.dumps(obj, indent=1, default=str, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def text(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")

    def close(self) -> None:
        for i, srv in enumerate(self.servers):
            srv.dump_journal(self.dir / "requests.jsonl")
            if srv.log_path is None:
                continue
            # One log per start: a restart case keeps the run before the kill.
            own = re.compile(rf"server-{re.escape(srv.root.name)}-(\d+)\.log")
            logs = sorted(
                (p for p in srv.log_path.parent.iterdir() if own.fullmatch(p.name)),
                key=lambda p: int(own.fullmatch(p.name)[1]),
            )
            for n, log in enumerate(logs):
                name = (
                    f"server-{i}.log" if len(logs) == 1 else f"server-{i}-start{n}.log"
                )
                self.text(name, _ANSI.sub("", log.read_text(errors="replace")))
        rerun = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
        self.text("rerun.txt", f"uv run pytest '{rerun}' -m acceptance -v\n")


def evidence_base(tmp_path: Path) -> Path:
    return Path(os.environ.get("PURGE_EVIDENCE_DIR", tmp_path / "evidence"))


def p0(root: Path, scopes: tuple[tuple[str, str], ...] = WORLD) -> dict[str, Any]:
    """Every table of every scope holds rows (else a purge proves nothing)."""
    counts = {f"{a}/{p}": scope_counts(root, a, p) for a, p in scopes}
    empty = {s: [k for k, n in c.items() if n == 0] for s, c in counts.items()}
    assert not any(empty.values()), f"P0: the seed left empty tables: {empty}"
    return counts


def purge(srv: PurgeServer, **body: Any) -> tuple[str, dict[str, Any]]:
    """request + confirm (202); the job id and the request's data."""
    req = srv.purge(**body)
    assert req.status_code == 200, req.text
    data = req.json()["data"]
    conf = srv.confirm(data["nonce"])
    assert conf.status_code == 202, conf.text
    return conf.json()["data"]["job_id"], data


def finish(srv: PurgeServer, job_id: str, *, timeout: float = 120) -> dict[str, Any]:
    """Poll every 0.5 s until terminal; the final job data."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = srv.job(job_id).json()["data"]
        if data["status"] in ("succeeded", "failed"):
            return data
        time.sleep(0.5)
    raise TimeoutError(f"purge {job_id} not terminal after {timeout}s")


def inbox_rows(root: Path, app: str, project: str) -> list[tuple[Any, ...]]:
    return rows(
        system_db(root),
        "SELECT request_id, status, payload_json, last_error FROM memorize_inbox"
        " WHERE app_id = ? AND project_id = ? ORDER BY id",
        app,
        project,
    )


@dataclass
class World:
    srv: PurgeServer
    fake: FakeUpstream
    log: SeedLog
    ev: Evidence
