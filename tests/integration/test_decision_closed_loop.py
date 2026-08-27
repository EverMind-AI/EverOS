"""Gate 7 — FakeLLM Decision closed loop.

Input 「我们决定使用 Rust 实现设备 Runtime。」 → extract_decision →
markdown daily-log → cascade Lance ``decision`` → keyword search
「设备 Runtime 为什么使用 Rust？」 recalls ``data.decisions``.

No real LLM / embedder. ``DecisionExtractor`` is the real class; only
atomic-fact / profile extractors are stubbed so their prompts cannot
steal FakeLLM turns. ``trigger_decision_clustering`` no-ops because
the suite's autouse fixture leaves embedding unavailable.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from pydantic import ValidationError
from sqlmodel import SQLModel

from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    decision_repo,
    dispose_connection,
    ensure_business_indexes,
)
from everos.infra.persistence.sqlite import (
    dispose_engine,
    get_engine,
    md_change_state_repo,
)
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator
from everos.memory.get import GetMemoryType, GetRequest
from everos.memory.search import SearchMethod, SearchRequest
from everos.service.memorize import memorize

_OWNER = "u_alice"
_SESSION = "s_decision_loop"
_USER_UTTERANCE = "我们决定使用 Rust 实现设备 Runtime。"
_SEARCH_QUERY = "设备 Runtime 为什么使用 Rust？"

_DECISION_JSON = json.dumps(
    {
        "decisions": [
            {
                "title": "Device Runtime language",
                "decision": "Use Rust for the device Runtime.",
                "reason": "Rust gives stable, low-overhead device Runtime.",
                "impact": "Device capabilities talk to the Agent Runtime over APIs.",
                "tags": ["architecture", "runtime"],
            }
        ]
    }
)


def _boundary_json() -> str:
    return json.dumps({"reasoning": "test", "boundaries": [], "should_wait": False})


def _episode_json() -> str:
    return json.dumps(
        {
            "title": "Device Runtime language choice",
            "content": "The team chose Rust for the device Runtime.",
            "summary": "Rust for device Runtime.",
        }
    )


def _make_fake_llm() -> FakeLLMClient:
    """Dispatch by prompt fingerprint so episode / decision never share a queue."""

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        prompt = messages[0].content if messages else ""
        text = prompt.lower() if isinstance(prompt, str) else ""
        if "committed decisions" in text:
            return ChatResponse(content=_DECISION_JSON, model="fake")
        if "boundaries" in text or "should_wait" in text:
            return ChatResponse(content=_boundary_json(), model="fake")
        return ChatResponse(content=_episode_json(), model="fake")

    return FakeLLMClient(handler=handler)


def _decision_md_files(root: Path) -> list[Path]:
    base = root / "default_app" / "default_project" / "users" / _OWNER / "decisions"
    if not base.is_dir():
        return []
    return sorted(base.glob("decision-*.md"))


@pytest_asyncio.fixture
async def closed_loop_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """Tmp memory root + OME + cascade; FakeLLM; real DecisionExtractor."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")
    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")

    from everos.config import load_settings

    load_settings.cache_clear()

    from everos.core.persistence.lancedb.repository import LanceRepoBase

    LanceRepoBase._reset_locks_for_tests()

    await dispose_connection()
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await ensure_business_indexes()

    svc = importlib.import_module("everos.service.memorize")
    search_svc = importlib.import_module("everos.service.search")
    get_svc = importlib.import_module("everos.service.get")
    client_mod = importlib.import_module("everos.component.llm.client")
    dc_mod = importlib.import_module("everos.memory.strategies.extract_decision")
    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    prof_mod = importlib.import_module("everos.memory.strategies.extract_user_profile")

    for attr in (
        "_episode_writer",
        "_prompt_loader",
        "_user_pipeline",
        "_agent_pipeline",
        "_ome_engine",
    ):
        monkeypatch.setattr(svc, attr, None, raising=False)
    monkeypatch.setattr(search_svc, "_manager", None, raising=False)
    monkeypatch.setattr(get_svc, "_manager", None, raising=False)
    monkeypatch.setattr(client_mod, "_llm_client", _make_fake_llm(), raising=False)
    monkeypatch.setattr(dc_mod, "_writer", None, raising=False)
    monkeypatch.setattr(af_mod, "_writer", None, raising=False)
    monkeypatch.setattr(prof_mod, "_writer", None, raising=False)
    monkeypatch.setattr(prof_mod, "_reader", None, raising=False)
    monkeypatch.setattr(prof_mod, "PROFILE_MIN_MEMCELLS", 99, raising=False)
    monkeypatch.setattr(
        af_mod,
        "AtomicFactExtractor",
        lambda *a, **k: type(
            "M",
            (),
            {"aextract_from_text": AsyncMock(return_value=[])},
        )(),
    )

    ome = svc._get_engine()
    await ome.start()

    cascade = CascadeOrchestrator(
        memory_root=MemoryRoot.resolve(),
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=0.5,
            worker_batch_size=10,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )
    await cascade.start()
    await asyncio.sleep(0.3)

    try:
        yield tmp_path
    finally:
        await cascade.stop()
        await ome.stop()
        await dispose_connection()
        await dispose_engine()
        monkeypatch.setattr(search_svc, "_manager", None, raising=False)
        monkeypatch.setattr(get_svc, "_manager", None, raising=False)


async def _poll(condition, *, deadline: float = 20.0) -> Any:  # type: ignore[no-untyped-def]
    async with asyncio.timeout(deadline):
        while True:
            result = (
                await condition()
                if asyncio.iscoroutinefunction(condition)
                else condition()
            )
            if result:
                return result
            await asyncio.sleep(0.05)


async def _wait_decision_md(root: Path) -> Path:
    def _ready() -> Path | None:
        files = _decision_md_files(root)
        for path in files:
            body = path.read_text(encoding="utf-8")
            if "dc_" in body and "Rust" in body and "### Title" in body:
                return path
        return None

    found = await _poll(_ready)
    assert isinstance(found, Path)
    return found


async def _wait_cascade_done(md_path: str) -> None:
    async def _done() -> bool:
        row = await md_change_state_repo.get_by_id(md_path)
        if row is not None and row.status == "failed":
            raise AssertionError(f"cascade failed for {md_path}: {row.error}")
        return row is not None and row.status == "done" and row.error is None

    await _poll(_done)


async def test_extract_md_cascade_search_recalls_rust_runtime(
    closed_loop_env: Path,
) -> None:
    """Design §12.2: extract → md → cascade → search recalls the Decision."""
    result = await memorize(
        {
            "session_id": _SESSION,
            "messages": [
                {
                    "sender_id": _OWNER,
                    "role": "user",
                    "content": _USER_UTTERANCE,
                    "timestamp": 1_700_000_000_000,
                },
                {
                    "sender_id": "assistant",
                    "role": "assistant",
                    "content": "Agreed. Rust on the device Runtime.",
                    "timestamp": 1_700_000_001_000,
                },
            ],
        },
        is_final=True,
    )
    assert result.status == "extracted"

    md_file = await _wait_decision_md(closed_loop_env)
    body = md_file.read_text(encoding="utf-8")
    assert "dc_" in body
    assert "Device Runtime language" in body
    assert "Use Rust for the device Runtime." in body
    assert "**parent_type**: memcell" in body
    assert f"**owner_id**: {_OWNER}" in body

    rel = md_file.relative_to(closed_loop_env).as_posix()
    await _wait_cascade_done(rel)

    rows = await decision_repo.find_by_owner(_OWNER)
    assert len(rows) >= 1
    hit = next(r for r in rows if "Rust" in r.decision)
    assert hit.entry_id.startswith("dc_")
    assert hit.deprecated_by is None
    assert hit.parent_type == "memcell"
    assert "Runtime" in hit.decision

    from everos.service.get import get as get_memory
    from everos.service.search import search as search_memory

    async def _keyword_hit():
        resp = await search_memory(
            SearchRequest(
                user_id=_OWNER,
                query=_SEARCH_QUERY,
                method=SearchMethod.KEYWORD,
                top_k=5,
            )
        )
        return resp if resp.data.decisions else None

    search_resp = await _poll(_keyword_hit, deadline=10.0)
    assert search_resp.data.episodes is not None
    assert search_resp.data.principles == []
    assert search_resp.data.profiles == []
    recalled = search_resp.data.decisions[0]
    blob = f"{recalled.title} {recalled.decision} {recalled.reason}"
    assert "Rust" in blob
    assert "Runtime" in blob

    episode_only = await search_memory(
        SearchRequest(
            user_id=_OWNER,
            query=_SEARCH_QUERY,
            method=SearchMethod.KEYWORD,
            kinds=["episode"],
            top_k=5,
        )
    )
    assert episode_only.data.decisions == []

    get_resp = await get_memory(
        GetRequest(user_id=_OWNER, memory_type=GetMemoryType.DECISION)
    )
    assert get_resp.data.decisions
    listed = get_resp.data.decisions[0]
    assert listed.decision == hit.decision
    assert listed.title == hit.title


def test_kinds_principle_still_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            user_id=_OWNER,
            query="x",
            kinds=["principle"],  # type: ignore[list-item]
        )


def test_get_memory_type_principle_still_rejected() -> None:
    with pytest.raises(ValidationError):
        GetRequest.model_validate({"user_id": _OWNER, "memory_type": "principle"})
