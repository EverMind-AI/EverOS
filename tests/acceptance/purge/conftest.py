"""Real ``everos server`` processes behind the proxy and the fake upstream."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.acceptance.memory_delete.conftest import _EVEROS, Server
from tests.acceptance.memory_delete.proxy import Proxy

from .fake_upstream import FakeUpstream

pytestmark = pytest.mark.acceptance

OME_TOML = """\
# Purge acceptance (H5): every table a purge clears gets seeded.
[strategies.extract_foresight]
enabled = true

[strategies.reflect_episodes]
enabled = true
"""


@dataclass
class PurgeServer(Server):
    """A :class:`Server` that journals every API call (evidence ring 1)."""

    fp_dir: Path = Path()
    journal: list[dict[str, Any]] = field(default_factory=list)

    def call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        version: str = "v2",
        timeout: float = 300,
        **kwargs: Any,
    ) -> httpx.Response:
        url = f"{self.base}/api/{version}{path}" if version else f"{self.base}{path}"
        sent = time.time()
        resp = httpx.request(method, url, json=body, timeout=timeout, **kwargs)
        self.journal.append(
            {
                "at": sent,
                "method": method,
                "path": url.removeprefix(self.base),
                "body": body if body is not None else kwargs.get("data"),
                "status": resp.status_code,
                "response": resp.text[:4000],
            }
        )
        return resp

    def post(
        self, path: str, body: dict[str, Any], *, timeout: float = 300
    ) -> httpx.Response:
        return self.call("POST", path, body, timeout=timeout)

    def purge(self, **body: Any) -> httpx.Response:
        return self.post("/memory/purge", body)

    def confirm(self, nonce: str) -> httpx.Response:
        return self.post("/memory/purge/confirm", {"nonce": nonce})

    def job(self, job_id: str) -> httpx.Response:
        return self.call("GET", f"/memory/purge/{job_id}")

    def wait_job(
        self, job_id: str, *, timeout: float = 120, until: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        """Poll the job until it reaches a terminal state (or one of ``until``)."""
        want = until or ("succeeded", "failed")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self.job(job_id).json()["data"]
            if data["status"] in want:
                return data
            time.sleep(0.2)
        raise TimeoutError(f"purge {job_id} never reached {want}")

    def dump_journal(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            for item in self.journal:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")


@pytest.fixture
def fake() -> Iterator[FakeUpstream]:
    upstream = FakeUpstream()
    upstream.start()
    yield upstream
    upstream.stop()


@pytest.fixture
def purge_proxy(fake: FakeUpstream) -> Iterator[Proxy]:
    p = Proxy(
        {
            "llm": f"{fake.base}/llm",
            "emb": f"{fake.base}/emb",
            "rerank": f"{fake.base}/rerank",
        },
        {"llm": "fake", "emb": "fake", "rerank": "fake"},
    )
    p.start()
    yield p
    p.stop()


@pytest.fixture
def make_purge_server(
    tmp_path: Path, purge_proxy: Proxy, fake: FakeUpstream
) -> Iterator[Callable[..., PurgeServer]]:
    """``make_purge_server(name, start=True, **env)``: a server over
    ``tmp_path/name`` with Foresight and Reflection on (H5); not started when
    ``start=False`` (seed the root first, then ``.start()``)."""
    made: list[PurgeServer] = []

    def make(
        name: str = "root",
        *,
        start: bool = True,
        everos: Path = _EVEROS,
        **env: str,
    ) -> PurgeServer:
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        fp_dir = tmp_path / f"fp-{name}"
        fp_dir.mkdir(exist_ok=True)
        srv = PurgeServer(root=root, proxy=purge_proxy, everos=everos, fp_dir=fp_dir)
        srv.env_extra |= {
            "EVEROS_EMBEDDING__DIM": str(fake.dim),
            "EVEROS_RERANK__PROVIDER": "vllm",
            "EVEROS_RERANK__MODEL": "fake-rerank",
            "EVEROS_RERANK__API_KEY": "via-proxy",
            "EVEROS_RERANK__BASE_URL": f"{purge_proxy.base}/rerank",
            "EVEROS_TEST_FAILPOINT_DIR": str(fp_dir),
        } | env
        if not (root / "everos.toml").exists():
            subprocess.run(
                [str(everos), "init", "--root", str(root)],
                env=srv.env(),
                check=True,
                capture_output=True,
            )
        ome_toml = root / "ome.toml"  # `everos init` writes an all-comment one
        text = ome_toml.read_text(encoding="utf-8") if ome_toml.exists() else ""
        if OME_TOML not in text:
            ome_toml.write_text(text + "\n" + OME_TOML, encoding="utf-8")
        made.append(srv)
        return srv.start() if start else srv  # type: ignore[return-value]

    yield make
    for srv in made:
        srv.stop()


@pytest.fixture
def make_world(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    make_purge_server: Callable[..., PurgeServer],
    fake: FakeUpstream,
) -> Iterator[Callable[..., Any]]:
    """``make_world(case, **env)``: a started server over the seeded world
    (four scopes, P0 checked), with evidence kept under ``case``."""
    from .journey import Evidence, World, evidence_base, p0
    from .observe import wait_idle
    from .seed import WORLD, seed_purge_world

    made: list[Evidence] = []

    def make(case: str, *, seed: bool = True, **env: str) -> World:
        srv = make_purge_server(**env)
        ev = Evidence(case, evidence_base(tmp_path), [srv])
        made.append(ev)
        log = seed_purge_world(srv, fake, WORLD) if seed else None
        if seed:
            wait_idle(srv.base, srv.root, timeout=300)
            ev.json("p0", p0(srv.root))
        return World(srv, fake, log, ev)  # type: ignore[arg-type]

    yield make
    for ev in made:
        ev.close()
