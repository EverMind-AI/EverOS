"""T7.2, TB4.1, T24.1 (a): the Milvus index backend."""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from .conftest import chat, free_port
from .observe import wait_converged

pytestmark = pytest.mark.acceptance

_ETCD = """listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
"""


@pytest.fixture
def milvus(tmp_path: Path) -> Iterator[dict[str, str]]:
    """A Milvus 2.6.22 standalone container, as CI starts it, on free ports."""
    etcd = tmp_path / "embedEtcd.yaml"
    etcd.write_text(_ETCD)
    name = f"everos-acc-milvus-{uuid.uuid4().hex[:8]}"
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
                if (
                    httpx.get(
                        f"http://127.0.0.1:{health}/healthz", timeout=2
                    ).status_code
                    == 200
                ):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("milvus never became healthy")
        yield {
            "uri": f"http://127.0.0.1:{grpc}",
            "name": name,
            "prefix": f"acc_{uuid.uuid4().hex[:8]}",
        }
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _env(m: dict[str, str]) -> dict[str, str]:
    return {
        "EVEROS_INDEX__BACKEND": "milvus",
        "EVEROS_MILVUS__URI": m["uri"],
        "EVEROS_MILVUS__COLLECTION_PREFIX": m["prefix"],
    }


def _milvus_ids(m: dict[str, str], table: str) -> set[str]:
    """Primary ids of the ``table`` collection, read straight from Milvus."""
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
        # Strong: the server writes through its own client; Session would let
        # this one read from before the server's last delete.
        found = client.query(
            names[0],
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
            consistency_level="Strong",
        )
        return {row["id"] for row in found}
    finally:
        client.close()


def _episode_ids(m: dict[str, str], user: str) -> set[str]:
    return {i for i in _milvus_ids(m, "episode") if i.startswith(f"{user}_")}


def test_t7_2_index_rows_go_with_the_response_on_milvus(
    make_server: Any, milvus: dict[str, str]
) -> None:
    """A7 on Milvus: projection stopped, the deletion removes the rows."""
    srv = make_server()
    srv.start(**_env(milvus))
    assert srv.add("s1", chat("u1", "civet31")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_converged(srv.base)
    ids = _episode_ids(milvus, "u1")
    assert ids, "positive control: indexed in Milvus"
    assert srv.post("/cascade/quiesce", {}).status_code == 200

    assert srv.delete(session_id="s1").status_code == 200

    assert not _episode_ids(milvus, "u1") & ids


def test_tb4_1_by_id_on_milvus(make_server: Any, milvus: dict[str, str]) -> None:
    """B4 on Milvus: the episode's and its facts' rows go with the response."""
    srv = make_server()
    srv.start(**_env(milvus))
    assert srv.add("s1", chat("u1", "genet52")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_converged(srv.base)
    ids = _episode_ids(milvus, "u1")
    assert len(ids) == 1, ids  # so every u1 fact is this episode's
    facts = {i for i in _milvus_ids(milvus, "atomic_fact") if i.startswith("u1_")}
    assert facts, "positive control: the facts are indexed in Milvus"
    assert srv.post("/cascade/quiesce", {}).status_code == 200

    (target,) = ids
    assert srv.delete_by_ids([target]).status_code == 200

    assert target not in _episode_ids(milvus, "u1")
    assert not _milvus_ids(milvus, "atomic_fact") & facts


class _Gate:
    """A TCP relay to Milvus that can stop moving bytes (connections stay
    open): the backend looks stuck, as a network partition makes it."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.port = free_port()
        self.loop = asyncio.new_event_loop()
        self.open: asyncio.Event | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.open = asyncio.Event()
        self.open.set()
        server = self.loop.run_until_complete(
            asyncio.start_server(self._serve, "127.0.0.1", self.port)
        )
        self.ready.set()
        try:
            self.loop.run_forever()
        finally:
            server.close()

    async def _pump(self, reader: Any, writer: Any) -> None:
        try:
            while data := await reader.read(65536):
                assert self.open is not None
                await self.open.wait()
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def _serve(self, reader: Any, writer: Any) -> None:
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", self.target)
        await asyncio.gather(
            self._pump(reader, up_writer), self._pump(up_reader, writer)
        )

    def start(self) -> _Gate:
        self.thread.start()
        self.ready.wait(10)
        return self

    def block(self) -> None:
        assert self.open is not None
        self.loop.call_soon_threadsafe(self.open.clear)

    def unblock(self) -> None:
        assert self.open is not None
        self.loop.call_soon_threadsafe(self.open.set)

    def stop(self) -> None:
        self.unblock()
        self.loop.call_soon_threadsafe(self.loop.stop)


def test_t24_1_a_unreachable_index_still_200_and_caught_up(
    make_server: Any, milvus: dict[str, str]
) -> None:
    """A24 (a): Milvus unreachable during the deletion -> 200 within the
    bound, one failure line per table; reachable again, the rows go within
    60 s (deviation 9: a relay stops moving bytes instead of docker pause,
    which kills Milvus on this host)."""
    gate = _Gate(int(milvus["uri"].rsplit(":", 1)[1])).start()
    srv = make_server()
    srv.start(**_env(milvus) | {"EVEROS_MILVUS__URI": f"http://127.0.0.1:{gate.port}"})
    assert srv.add("s1", chat("u1", "fossa63")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_converged(srv.base)
    ids = _episode_ids(milvus, "u1")
    assert ids
    gate.block()
    try:
        started = time.monotonic()
        resp = srv.delete(session_id="s1")
        took = time.monotonic() - started
    finally:
        gate.unblock()

    assert resp.status_code == 200, resp.text
    assert took <= 15 * 3 + 1, took  # episode, atomic_fact, foresight tables
    deadline = time.monotonic() + 60
    left = ids
    while time.monotonic() < deadline:
        try:
            left = _episode_ids(milvus, "u1") & ids
        except Exception:  # this test's own client reconnecting after unpause
            left = ids
        if not left:
            break
        time.sleep(2)
    assert not left
    srv.stop()
    failed = [
        line
        for line in srv.log_text().splitlines()
        if "memory_delete_index_erase_failed" in line
    ]
    assert failed, "the unreachable index was logged"
    assert all("table" in line for line in failed)
    gate.stop()
