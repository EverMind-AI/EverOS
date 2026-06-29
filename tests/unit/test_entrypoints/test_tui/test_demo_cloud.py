"""EverOS Cloud demo client contracts."""

from __future__ import annotations

import urllib.error

import pytest

from everos.entrypoints.tui.demo import cloud


def test_resolve_cloud_base_url_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.delenv(cloud.CLOUD_DEMO_SERVER_URL_ENV, raising=False)
    assert (
        cloud.resolve_cloud_base_url(cloud.LIVE_DEMO_SERVER_URL)
        == cloud.CLOUD_API_BASE_URL
    )

    monkeypatch.setenv(cloud.CLOUD_DEMO_SERVER_URL_ENV, "https://env.test")
    assert (
        cloud.resolve_cloud_base_url(cloud.LIVE_DEMO_SERVER_URL) == "https://env.test"
    )

    assert (
        cloud.resolve_cloud_base_url("https://explicit.test") == "https://explicit.test"
    )


def test_resolve_keys_read_their_env_vars(monkeypatch) -> None:
    monkeypatch.setenv(cloud.CLOUD_DEMO_KEY_ENV, "demo-key")
    monkeypatch.setenv(cloud.CLOUD_USER_KEY_ENV, "user-key")
    assert cloud.resolve_demo_key() == "demo-key"
    assert cloud.resolve_user_key() == "user-key"


def test_new_demo_identity_is_unique_and_paired() -> None:
    session_a, user_a = cloud.new_demo_identity()
    session_b, user_b = cloud.new_demo_identity()

    assert session_a != session_b
    assert user_a != user_b
    assert session_a.startswith("everos-demo-")
    assert user_a.startswith("everos_demo_")


def test_add_memory_posts_messages_and_returns_task_id() -> None:
    calls: list[tuple[str, str, dict[str, object] | None, str | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        api_key: str | None = None,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        calls.append((method, path, json_body, api_key))
        return {"data": {"task_id": "task-123", "status": "queued"}}

    task_id = cloud.add_memory(
        "我喜欢吃杨梅",
        base_url="https://api.test",
        session_id="everos-demo-abc",
        user_id="everos_demo_abc",
        api_key="k-1",
        request_json=fake_request,
    )

    assert task_id == "task-123"
    method, path, body, api_key = calls[0]
    assert (method, path) == ("POST", "/api/v1/memories")
    assert api_key == "k-1"
    assert body["user_id"] == "everos_demo_abc"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "我喜欢吃杨梅"


def test_flush_memory_forces_extraction() -> None:
    bodies: list[dict[str, object] | None] = []

    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        api_key: str | None = None,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        bodies.append(json_body)
        assert path == "/api/v1/memories/flush"
        return {"data": {"status": "extracted"}}

    cloud.flush_memory(
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
    )

    assert bodies[0]["force"] is True
    assert bodies[0]["user_id"] == "u"


def test_wait_task_succeeds_and_fails() -> None:
    def ok(*_: object, **__: object) -> dict[str, object]:
        return {"task_id": "t", "status": "success"}

    cloud.wait_task("t", base_url="https://api.test", api_key="k", request_json=ok)

    def failed(*_: object, **__: object) -> dict[str, object]:
        return {"task_id": "t", "status": "failed"}

    with pytest.raises(cloud.CloudDemoError):
        cloud.wait_task(
            "t",
            base_url="https://api.test",
            api_key="k",
            request_json=failed,
            attempts=1,
        )


def test_search_recall_parses_atomic_fact_and_score() -> None:
    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        api_key: str | None = None,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        assert path == "/api/v1/memories/search"
        assert json_body["filters"] == {"user_id": "everos_demo_abc"}
        return {
            "data": {
                "episodes": [
                    {
                        "id": "ep1",
                        "summary": "long summary",
                        "episode": "long episode text",
                        "score": None,
                        "atomic_facts": [
                            {
                                "id": "af1",
                                "atomic_fact": "You like Yangmei.",
                                "score": 0.57,
                            }
                        ],
                    }
                ]
            }
        }

    story = cloud.search_recall(
        "我喜欢吃杨梅",
        "我喜欢吃什么",
        base_url="https://api.test",
        user_id="everos_demo_abc",
        api_key="k",
        request_json=fake_request,
    )

    assert story is not None
    assert story.owner == "everos_demo_abc"
    assert story.answer == "You like Yangmei."  # from atomic_fact, not summary
    assert story.score == 0.57  # fact-level score (episode score is null)
    assert story.source_filename == "episode:ep1"


def test_search_recall_returns_none_on_miss() -> None:
    def fake_request(*_: object, **__: object) -> dict[str, object]:
        return {"data": {"episodes": []}}

    story = cloud.search_recall(
        "m",
        "q",
        base_url="https://api.test",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        search_attempts=2,
        search_interval_seconds=0.0,
    )

    assert story is None


def test_request_json_sets_bearer_header(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    class FakeResp:
        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *_: object) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        captured["auth"] = req.headers.get("Authorization")  # type: ignore[attr-defined]
        return FakeResp()

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_urlopen)
    cloud._request_json(
        "GET", "/x", base_url="https://api.test", api_key="abc", timeout_seconds=1.0
    )
    assert captured["auth"] == "Bearer abc"


def test_request_json_maps_401_and_429(monkeypatch) -> None:
    def boom(code: int):
        def _raise(*_: object, **__: object) -> object:
            raise urllib.error.HTTPError("https://api.test", code, "x", {}, None)

        return _raise

    monkeypatch.setattr(cloud.urllib.request, "urlopen", boom(401))
    with pytest.raises(cloud.CloudAuthError):
        cloud._request_json(
            "GET", "/x", base_url="https://api.test", timeout_seconds=1.0
        )

    monkeypatch.setattr(cloud.urllib.request, "urlopen", boom(429))
    with pytest.raises(cloud.CloudQuotaError):
        cloud._request_json(
            "GET", "/x", base_url="https://api.test", timeout_seconds=1.0
        )
