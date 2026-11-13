"""T14.1: a purge on the Milvus index backend deletes the range's rows only."""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import tests.acceptance.memory_delete.test_milvus as acc_milvus
from tests.acceptance.memory_delete.conftest import free_port

from .journey import Evidence, evidence_base, finish, purge
from .observe import INDEX_TABLES, wait_idle
from .seed import seed_purge_world

pytestmark = pytest.mark.acceptance


@pytest.fixture
def milvus(tmp_path: Path) -> Iterator[dict[str, str]]:
    """A Milvus 2.6.22 standalone container (as the deletion suite starts it),
    named ``everos-purge-*`` so a leftover is easy to find and remove."""
    etcd = tmp_path / "embedEtcd.yaml"
    etcd.write_text(acc_milvus._ETCD)
    name = f"everos-purge-milvus-{uuid.uuid4().hex[:8]}"
    grpc, health = free_port(), free_port()
    subprocess.run(
        [
            "docker", "run", "-d", "--name", name,
            "--security-opt", "seccomp:unconfined",
            "-e", "ETCD_USE_EMBED=true",
            "-e", "ETCD_DATA_DIR=/var/lib/milvus/etcd",
            "-e", "ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml",
            "-e", "COMMON_STORAGETYPE=local",
            "-e", "DEPLOY_MODE=STANDALONE",
            "-v", f"{etcd}:/milvus/configs/embedEtcd.yaml",
            "-p", f"{grpc}:19530", "-p", f"{health}:9091",
            "milvusdb/milvus:v2.6.22", "milvus", "run", "standalone",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                url = f"http://127.0.0.1:{health}/healthz"
                if httpx.get(url, timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("milvus never became healthy")
        yield {
            "uri": f"http://127.0.0.1:{grpc}",
            "prefix": f"pg_{uuid.uuid4().hex[:8]}",
        }
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _ids(m: dict[str, str], table: str, app: str, project: str) -> set[str]:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=m["uri"])
    try:
        names = [
            c
            for c in client.list_collections()
            if c.startswith(m["prefix"]) and c.endswith(table)
        ]
        if not names:
            return set()
        found = client.query(
            names[0],
            filter=f'app_id == "{app}" and project_id == "{project}"',
            output_fields=["id"],
            limit=16384,
            consistency_level="Strong",
        )
        return {row["id"] for row in found}
    finally:
        client.close()


def _all(m: dict[str, str], app: str, project: str) -> dict[str, set[str]]:
    return {t: _ids(m, t, app, project) for t in INDEX_TABLES}


def test_t14_1_milvus_purge_deletes_only_the_range(
    make_purge_server: Any, fake: Any, milvus: dict[str, str], tmp_path: Path
) -> None:
    srv = make_purge_server(**acc_milvus._env(milvus))
    ev = Evidence("T14.1", evidence_base(tmp_path), [srv])
    try:
        seed_purge_world(srv, fake, (("a1", "p1"), ("a1", "p2")))
        wait_idle(srv.base, srv.root, timeout=300)
        s_before, sib_before = _all(milvus, "a1", "p1"), _all(milvus, "a1", "p2")
        assert all(s_before.values()) and all(sib_before.values()), "P0 on Milvus"
        job_id, request = purge(srv, scope="project", app_id="a1", project_id="p1")
        assert request["summary"]["milvus_total"] is not None
        assert finish(srv, job_id)["status"] == "succeeded"
        wait_idle(srv.base, srv.root, timeout=300)
        s_after, sib_after = _all(milvus, "a1", "p1"), _all(milvus, "a1", "p2")
        ev.json(
            "ids",
            {
                "s_before": {k: len(v) for k, v in s_before.items()},
                "s_after": {k: len(v) for k, v in s_after.items()},
                "sibling_unchanged": sib_after == sib_before,
            },
        )
        assert all(not v for v in s_after.values()), s_after
        assert sib_after == sib_before
        # The whole root: delete_all's explicit always-true filter on 2.6.22.
        job_id, _ = purge(srv, scope="space")
        assert finish(srv, job_id)["status"] == "succeeded"
        assert all(not v for v in _all(milvus, "a1", "p2").values())
    finally:
        ev.close()
