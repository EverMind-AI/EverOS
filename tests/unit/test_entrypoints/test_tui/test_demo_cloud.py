"""EverOS hosted-demo cloud client contracts."""

from __future__ import annotations

import urllib.error

import pytest

from everos.entrypoints.tui.demo import cloud


def test_resolve_cloud_base_url_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.delenv(cloud.CLOUD_DEMO_SERVER_URL_ENV, raising=False)
    assert (
        cloud.resolve_cloud_base_url(cloud.LIVE_DEMO_SERVER_URL)
        == cloud.DEFAULT_CLOUD_DEMO_SERVER_URL
    )

    monkeypatch.setenv(cloud.CLOUD_DEMO_SERVER_URL_ENV, "https://env.test")
    assert (
        cloud.resolve_cloud_base_url(cloud.LIVE_DEMO_SERVER_URL) == "https://env.test"
    )

    # An explicit --server-url always wins over the env default.
    assert (
        cloud.resolve_cloud_base_url("https://explicit.test") == "https://explicit.test"
    )


def test_new_demo_identity_is_unique_and_paired() -> None:
    session_a, user_a = cloud.new_demo_identity()
    session_b, user_b = cloud.new_demo_identity()

    assert session_a != session_b
    assert user_a != user_b
    assert session_a.startswith("everos-demo-")
    assert user_a.startswith("everos_demo_")


def test_check_health_raises_when_not_ok() -> None:
    def fake_request(*_: object, **__: object) -> dict[str, object]:
        return {"status": "degraded"}

    with pytest.raises(cloud.CloudDemoError):
        cloud.check_health(base_url="http://server.test", request_json=fake_request)


def test_add_memory_sends_isolated_identity() -> None:
    bodies: list[tuple[str, dict[str, object] | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        bodies.append((path, json_body))
        return {}

    cloud.add_memory(
        "我喜欢吃杨梅",
        base_url="http://server.test",
        session_id="everos-demo-abc",
        user_id="everos_demo_abc",
        request_json=fake_request,
    )

    path, body = bodies[0]
    assert path == "/api/v1/memory/add"
    assert body is not None
    assert body["session_id"] == "everos-demo-abc"
    assert body["messages"][0]["sender_id"] == "everos_demo_abc"
    assert body["messages"][0]["content"] == "我喜欢吃杨梅"


def test_search_recall_returns_story_with_real_score() -> None:
    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        if path == "/api/v1/memory/search":
            return {
                "data": {
                    "episodes": [
                        {
                            "id": "ep1",
                            "summary": "You like Yangmei.",
                            "score": 0.41,
                            "atomic_facts": [
                                {
                                    "id": "af1",
                                    "content": "You like Yangmei.",
                                    "score": 0.87,
                                }
                            ],
                        }
                    ]
                }
            }
        return {}

    story = cloud.search_recall(
        "我喜欢吃杨梅",
        "我喜欢吃什么",
        base_url="http://server.test",
        user_id="everos_demo_abc",
        request_json=fake_request,
    )

    assert story is not None
    assert story.owner == "everos_demo_abc"
    assert story.query == "我喜欢吃什么"
    assert story.answer == "You like Yangmei."
    assert story.score == 0.87  # prefers the top fact's score


def test_search_recall_returns_none_on_miss() -> None:
    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {"data": {"episodes": []}}

    story = cloud.search_recall(
        "m",
        "q",
        base_url="http://server.test",
        user_id="u",
        request_json=fake_request,
        search_attempts=2,
        search_interval_seconds=0.0,
    )

    assert story is None


def test_request_json_maps_429_to_quota_error(monkeypatch) -> None:
    def boom(*_: object, **__: object) -> object:
        raise urllib.error.HTTPError(
            "http://server.test", 429, "Too Many Requests", {}, None
        )

    monkeypatch.setattr(cloud.urllib.request, "urlopen", boom)

    with pytest.raises(cloud.CloudQuotaError):
        cloud._request_json(
            "GET", "/health", base_url="http://server.test", timeout_seconds=1.0
        )
