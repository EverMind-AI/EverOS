"""Contract tests for ``integrations/hermes/__init__.py`` (the provider).

Pins the ``EverosMemoryProvider`` lifecycle / tools / circuit-breaker
behaviour against a fake EverOS client. Hermes-only symbols
(``agent.memory_provider``, ``hermes_constants``, ``tools.registry``,
``utils``) are injected via ``sys.modules`` (see ``tests.helpers.
hermes_stub``) so the plugin bundle imports cleanly without a Hermes
runtime. No network, no ``respx`` / ``requests_mock`` — the fake client
records calls and returns canned ``SearchData`` / ``GetData`` /
``AddResponse`` / ``FlushResponse`` dicts.
"""

from __future__ import annotations

import contextlib
import json
import stat
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module", autouse=True)
def _hermes_stubs():
    """Inject Hermes stubs into ``sys.modules`` and import the plugin.

    The plugin's ``__init__.py`` imports ``agent.memory_provider``,
    ``hermes_constants``, ``tools.registry`` and ``utils`` — none of which
    are installable from this repo. We point them all at
    ``tests.helpers.hermes_stub`` (which carries every symbol the plugin
    references) and add the repo root to ``sys.path`` so
    ``integrations.hermes`` imports as a real package.
    """
    import tests.helpers.hermes_stub as stub

    mods = {
        "agent": types.ModuleType("agent"),
        "agent.memory_provider": stub,
        "hermes_constants": stub,
        "tools": types.ModuleType("tools"),
        "tools.registry": stub,
        "utils": stub,
    }
    mods["agent"].memory_provider = stub  # type: ignore[attr-defined]
    mods["tools"].registry = stub  # type: ignore[attr-defined]
    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)
    sys.path.insert(0, str(_REPO_ROOT))
    try:
        import importlib

        plugin = importlib.import_module("integrations.hermes")
        yield plugin
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(_REPO_ROOT))


# ── canned API response shapes ──────────────────────────────────────────────


def _empty_search_data() -> dict[str, Any]:
    return {
        "episodes": [],
        "profiles": [],
        "agent_cases": [],
        "agent_skills": [],
        "unprocessed_messages": [],
    }


def _search_data_with_episode() -> dict[str, Any]:
    ep = {
        "id": "e1",
        "user_id": "alice",
        "app_id": "default",
        "project_id": "default",
        "session_id": "sess-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "sender_ids": ["alice"],
        "summary": "sum",
        "subject": "subj",
        "episode": "Alice likes tea.",
        "type": "episode",
        "score": 0.9,
        "atomic_facts": [{"id": "f1", "content": "Alice likes tea.", "score": 0.9}],
    }
    return {**_empty_search_data(), "episodes": [ep]}


def _empty_get_data() -> dict[str, Any]:
    return {
        "episodes": [],
        "profiles": [],
        "agent_cases": [],
        "agent_skills": [],
        "total_count": 0,
        "count": 0,
    }


class FakeEverOSClient:
    """Recording stand-in for ``EverOSApiClient``.

    All methods return canned dicts; ``raise_on`` maps a method name to an
    error code that the method should raise as ``EverOSClientError``.
    """

    def __init__(
        self,
        *,
        search_data: dict[str, Any] | None = None,
        get_data: dict[str, Any] | None = None,
        add_response: dict[str, Any] | None = None,
        flush_response: dict[str, Any] | None = None,
        raise_on: dict[str, str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self._search_data = (
            search_data if search_data is not None else _empty_search_data()
        )
        self._get_data = get_data if get_data is not None else _empty_get_data()
        self._add_response = add_response or {
            "message_count": 1,
            "status": "accumulated",
        }
        self._flush_response = flush_response or {"status": "extracted"}
        self._raise_on = raise_on or {}

    def _maybe_raise(self, method: str) -> None:
        code = self._raise_on.get(method)
        if code is not None:
            from integrations.hermes._types import EverOSClientError

            raise EverOSClientError("boom", code=code)

    def add_messages(
        self,
        session_id: str,
        app_id: str,
        project_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._maybe_raise("add_messages")
        self.calls.append(
            (
                "add_messages",
                {
                    "session_id": session_id,
                    "app_id": app_id,
                    "project_id": project_id,
                    "messages": list(messages),
                },
            )
        )
        return self._add_response

    def flush_session(
        self, session_id: str, app_id: str, project_id: str
    ) -> dict[str, Any]:
        self._maybe_raise("flush_session")
        self.calls.append(
            (
                "flush_session",
                {
                    "session_id": session_id,
                    "app_id": app_id,
                    "project_id": project_id,
                },
            )
        )
        return self._flush_response

    def search(
        self,
        user_id: str | None,
        agent_id: str | None,
        app_id: str,
        project_id: str,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._maybe_raise("search")
        self.calls.append(
            (
                "search",
                {
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "app_id": app_id,
                    "project_id": project_id,
                    "query": query,
                    **kwargs,
                },
            )
        )
        return self._search_data

    def get(
        self,
        user_id: str | None,
        agent_id: str | None,
        app_id: str,
        project_id: str,
        memory_type: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._maybe_raise("get")
        self.calls.append(
            (
                "get",
                {
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "app_id": app_id,
                    "project_id": project_id,
                    "memory_type": memory_type,
                    **kwargs,
                },
            )
        )
        return self._get_data

    def close(self) -> None:
        self.closed = True
        self.calls.append(("close", {}))


@pytest.fixture
def plugin(_hermes_stubs):
    return _hermes_stubs


@pytest.fixture
def make_provider(plugin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a factory that builds an initialised provider + fake client."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("EVEROS_API_URL", raising=False)
    monkeypatch.delenv("EVEROS_USER_ID", raising=False)
    monkeypatch.delenv("EVEROS_AGENT_ID", raising=False)
    monkeypatch.delenv("EVEROS_MODE", raising=False)

    def _make(
        *,
        fake: FakeEverOSClient | None = None,
        config: dict[str, Any] | None = None,
        **init_kwargs: Any,
    ):
        if config is not None:
            (tmp_path / "everos.json").write_text(json.dumps(config))
        prov = plugin.EverosMemoryProvider()
        prov.initialize("sess-1", **init_kwargs)
        if fake is not None:
            prov._client = fake
        return prov

    return _make


# ── identity / availability ─────────────────────────────────────────────────


def test_name_and_availability(make_provider, plugin):
    prov = make_provider()
    assert prov.name == "everos"
    # Defaults carry a non-empty api_url → is_available() is True.
    assert prov.is_available() is True


def test_is_available_false_when_unconfigured(
    make_provider, plugin, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(plugin, "is_configured", lambda _cfg: False)
    prov = make_provider()
    assert prov.is_available() is False


# ── initialize ───────────────────────────────────────────────────────────────


def test_initialize_builds_state_from_config_and_kwargs(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(
        fake=fake,
        config={
            "app_id": "myapp",
            "project_id": "myproj",
            "agent_id": "myagent",
        },
        user_id="alice",
    )
    assert prov._session_id == "sess-1"
    assert prov._user_id == "alice"
    assert prov._agent_id == "myagent"
    assert prov._scope is not None
    assert prov._scope.app_id == "myapp"
    assert prov._scope.project_id == "myproj"
    assert prov._client is fake
    assert prov._init_error == ""
    prov.shutdown()


def test_initialize_bad_url_leaves_client_none(
    make_provider, plugin, monkeypatch: pytest.MonkeyPatch
):
    def _boom(_url: str):
        raise ValueError("bad url")

    monkeypatch.setattr(plugin, "EverOSApiClient", _boom)
    prov = make_provider()
    assert prov._client is None
    assert "bad url" in prov._init_error


# ── prefetch ────────────────────────────────────────────────────────────────


def test_prefetch_cache_hit_returns_formatted_context(make_provider):
    fake = FakeEverOSClient(search_data=_search_data_with_episode())
    prov = make_provider(fake=fake)
    # Prime the cache: simulate a completed prefetch for this query.
    prov._prefetch_query = "tea"
    prov._prefetch_result = "## EverOS Memory\ncached"
    prov._prefetch_done = True
    out = prov.prefetch("tea")
    assert out == "## EverOS Memory\ncached"
    # The worker must not have been started (cache hit).
    assert not any(c[0] == "search" for c in fake.calls)
    prov.shutdown()


def test_prefetch_first_turn_starts_worker_and_joins(make_provider):
    fake = FakeEverOSClient(search_data=_search_data_with_episode())
    prov = make_provider(fake=fake)
    out = prov.prefetch("tea")
    assert "tea" in out.lower() or "everos" in out.lower()
    assert any(c[0] == "search" for c in fake.calls)
    # The prefetch thread has been joined by prefetch() itself.
    assert prov._prefetch_thread is not None
    assert not prov._prefetch_thread.is_alive()
    prov.shutdown()


def test_prefetch_empty_results_returns_empty_string(make_provider):
    fake = FakeEverOSClient(search_data=_empty_search_data())
    prov = make_provider(fake=fake)
    assert prov.prefetch("nothing") == ""
    prov.shutdown()


def test_prefetch_breaker_open_returns_empty(make_provider):
    fake = FakeEverOSClient(search_data=_search_data_with_episode())
    prov = make_provider(fake=fake)
    prov._consecutive_failures = 5
    prov._breaker_open_until = _now_plus(1000.0)
    assert prov.prefetch("tea") == ""
    assert not any(c[0] == "search" for c in fake.calls)
    prov.shutdown()


# ── sync_turn ───────────────────────────────────────────────────────────────


def test_sync_turn_enqueues_user_then_assistant_with_adjacent_timestamps(
    make_provider,
):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov.sync_turn("hello", "hi there")
    thread = prov._sync_thread
    assert thread is not None
    thread.join(timeout=2.0)
    add_calls = [c for c in fake.calls if c[0] == "add_messages"]
    assert add_calls, "add_messages was not called"
    msgs = add_calls[-1][1]["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["timestamp"] - msgs[0]["timestamp"] == 1
    prov.shutdown()


def test_sync_turn_joins_previous_thread(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov.sync_turn("first", "a1")
    prov.sync_turn("second", "a2")
    thread = prov._sync_thread
    assert thread is not None
    thread.join(timeout=2.0)
    add_calls = [c for c in fake.calls if c[0] == "add_messages"]
    assert len(add_calls) == 2
    prov.shutdown()


def test_sync_turn_skips_when_breaker_open(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov._consecutive_failures = 5
    prov._breaker_open_until = _now_plus(1000.0)
    prov.sync_turn("x", "y")
    assert prov._sync_thread is None
    assert not any(c[0] == "add_messages" for c in fake.calls)
    prov.shutdown()


# ── on_session_end ──────────────────────────────────────────────────────────


def test_on_session_end_flushes_and_closes_client(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov.sync_turn("hi", "hello")
    prov.on_session_end([])
    assert any(c[0] == "flush_session" for c in fake.calls)
    assert fake.closed is True
    assert prov._client is None


# ── on_memory_write ─────────────────────────────────────────────────────────


def test_on_memory_write_mirrors_add_user(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov.on_memory_write("add", "user", "Alice likes tea.")
    mirror = prov._mirror_thread
    assert mirror is not None
    mirror.join(timeout=2.0)
    add_calls = [c for c in fake.calls if c[0] == "add_messages"]
    flush_calls = [c for c in fake.calls if c[0] == "flush_session"]
    assert add_calls and flush_calls
    assert add_calls[-1][1]["messages"][0]["content"] == "Alice likes tea."
    prov.shutdown()


def test_on_memory_write_skips_memory_target(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov.on_memory_write("add", "memory", "ignored")
    assert prov._mirror_thread is None
    assert not fake.calls
    prov.shutdown()


def test_on_memory_write_remove_is_noop(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov.on_memory_write("remove", "user", "x")
    assert prov._mirror_thread is None
    assert not fake.calls
    prov.shutdown()


# ── handle_tool_call ────────────────────────────────────────────────────────


def test_handle_tool_call_search_happy_path(make_provider):
    fake = FakeEverOSClient(search_data=_search_data_with_episode())
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_search", {"query": "tea"})
    payload = json.loads(out)
    assert "results" in payload
    assert payload["count"] == 1
    assert any(c[0] == "search" for c in fake.calls)
    prov.shutdown()


def test_handle_tool_call_list_happy_path(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_list", {"memory_type": "episode"})
    payload = json.loads(out)
    assert "results" in payload
    assert any(c[0] == "get" for c in fake.calls)
    prov.shutdown()


def test_handle_tool_call_add_happy_path(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_add", {"content": "a fact"})
    payload = json.loads(out)
    assert payload["result"] == "Fact stored."
    assert any(c[0] == "add_messages" for c in fake.calls)
    assert any(c[0] == "flush_session" for c in fake.calls)
    prov.shutdown()


def test_handle_tool_call_flush_happy_path(make_provider):
    fake = FakeEverOSClient(flush_response={"status": "extracted"})
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_flush", {})
    payload = json.loads(out)
    assert payload["status"] == "extracted"
    prov.shutdown()


def test_handle_tool_call_search_missing_query_returns_error(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_search", {})
    assert "error" in json.loads(out)
    prov.shutdown()


def test_handle_tool_call_add_missing_content_returns_error(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_add", {})
    assert "error" in json.loads(out)
    prov.shutdown()


def test_handle_tool_call_unknown_tool_returns_error(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_bogus", {})
    assert "error" in json.loads(out)
    prov.shutdown()


def test_handle_tool_call_backend_down_returns_error(make_provider):
    prov = make_provider()
    prov._client = None
    prov._init_error = "init failed"
    out = prov.handle_tool_call("everos_search", {"query": "x"})
    assert "error" in json.loads(out)


def test_handle_tool_call_list_transient_failure_trips_breaker(make_provider):
    fake = FakeEverOSClient(raise_on={"get": "EXTERNAL_SERVICE_UNAVAILABLE"})
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_list", {"memory_type": "episode"})
    payload = json.loads(out)
    assert "error" in payload
    assert prov._consecutive_failures == 1
    prov.shutdown()


def test_handle_tool_call_flush_transient_failure_trips_breaker(make_provider):
    fake = FakeEverOSClient(raise_on={"flush_session": "INTERNAL_ERROR"})
    prov = make_provider(fake=fake)
    out = prov.handle_tool_call("everos_flush", {})
    payload = json.loads(out)
    assert "error" in payload
    assert prov._consecutive_failures == 1
    prov.shutdown()


# ── get_tool_schemas / system_prompt_block ─────────────────────────────────


def test_get_tool_schemas_openai_shape(make_provider):
    prov = make_provider()
    schemas = prov.get_tool_schemas()
    assert len(schemas) == 4
    names = {s["name"] for s in schemas}
    assert names == {"everos_search", "everos_list", "everos_add", "everos_flush"}
    for schema in schemas:
        assert schema["parameters"]["type"] == "object"
        assert "properties" in schema["parameters"]
        assert "description" in schema
    prov.shutdown()


def test_system_prompt_block_mentions_everos_search(make_provider):
    prov = make_provider()
    block = prov.system_prompt_block()
    assert "everos" in block.lower()
    assert "everos_search" in block
    prov.shutdown()


# ── circuit breaker ─────────────────────────────────────────────────────────


def test_breaker_opens_after_five_transient_failures(make_provider, plugin):
    prov = make_provider()
    for _ in range(5):
        prov._record_failure()
    assert prov._is_breaker_open() is True
    prov.shutdown()


def test_breaker_skips_calls_when_open(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov._consecutive_failures = 5
    prov._breaker_open_until = _now_plus(1000.0)
    prov.sync_turn("x", "y")
    assert not any(c[0] == "add_messages" for c in fake.calls)
    prov.shutdown()


def test_breaker_recovers_after_cooldown(make_provider):
    fake = FakeEverOSClient()
    prov = make_provider(fake=fake)
    prov._consecutive_failures = 5
    prov._breaker_open_until = _now_minus(1.0)  # cooldown elapsed
    assert prov._is_breaker_open() is False
    assert prov._consecutive_failures == 0
    prov.sync_turn("x", "y")
    thread = prov._sync_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert any(c[0] == "add_messages" for c in fake.calls)
    prov.shutdown()


def test_invalid_input_does_not_trip_breaker(make_provider, plugin):

    fake = FakeEverOSClient(raise_on={"add_messages": "INVALID_INPUT"})
    prov = make_provider(fake=fake)
    prov.sync_turn("x", "y")
    thread = prov._sync_thread
    assert thread is not None
    thread.join(timeout=2.0)
    # Client errors must not count toward the breaker.
    assert prov._consecutive_failures == 0
    assert prov._is_breaker_open() is False
    prov.shutdown()


def test_is_transient_classification(plugin):
    assert plugin.EverosMemoryProvider._is_transient("INTERNAL_ERROR") is True
    assert plugin.EverosMemoryProvider._is_transient("CLIENT_CLOSED") is True
    assert plugin.EverosMemoryProvider._is_transient("INVALID_INPUT") is False


# ── backup_paths ────────────────────────────────────────────────────────────


def test_backup_paths_returns_everos_root_when_present(
    make_provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import pathlib

    everos_dir = tmp_path / ".everos"
    everos_dir.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    prov = make_provider()
    assert prov.backup_paths() == [str(everos_dir)]
    prov.shutdown()


def test_backup_paths_empty_when_absent(
    make_provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    prov = make_provider()
    assert prov.backup_paths() == []
    prov.shutdown()


# ── shutdown ────────────────────────────────────────────────────────────────


def test_shutdown_joins_threads_and_closes_client(make_provider):
    fake = FakeEverOSClient(search_data=_search_data_with_episode())
    prov = make_provider(fake=fake)
    prov.sync_turn("hi", "hello")
    prov.prefetch("tea")
    prov.shutdown()
    assert fake.closed is True
    assert prov._client is None


# ── config schema / save_config / post_setup ───────────────────────────────


def test_get_config_schema_has_seven_entries(make_provider):
    prov = make_provider()
    schema = prov.get_config_schema()
    assert len(schema) == 7
    for entry in schema:
        assert "key" in entry
        assert "description" in entry
    keys = {e["key"] for e in schema}
    assert "api_url" in keys
    assert "agent_track_enabled" in keys
    prov.shutdown()


def test_save_config_writes_mode_0600(make_provider, tmp_path: Path):
    prov = make_provider()
    prov.save_config({"api_url": "http://x"}, str(tmp_path))
    path = tmp_path / "everos.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["api_url"] == "http://x"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    prov.shutdown()


def test_post_setup_delegates_to_setup_module(
    make_provider, plugin, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[Any, ...]] = []

    def _spy(hermes_home: Path, config: dict[str, Any], *, interactive: bool = True):
        calls.append((hermes_home, config, interactive))
        return config

    monkeypatch.setattr(plugin, "_run_setup", _spy)
    prov = make_provider()
    cfg = {"mode": "platform"}
    prov.post_setup("/tmp/hermes-home", cfg)
    assert len(calls) == 1
    assert calls[0][0] == Path("/tmp/hermes-home")
    assert calls[0][1] is cfg
    assert calls[0][2] is True
    prov.shutdown()


# ── helpers ─────────────────────────────────────────────────────────────────


def _now_plus(seconds: float) -> float:
    import time

    return time.monotonic() + seconds


def _now_minus(seconds: float) -> float:
    import time

    return time.monotonic() - seconds
