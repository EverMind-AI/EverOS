"""EverOS hosted-demo cloud client contracts."""

from __future__ import annotations

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


def test_recall_round_runs_real_flow_with_isolated_identity() -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        calls.append((method, path, json_body))
        assert base_url == "http://server.test"
        if path == "/health":
            return {"status": "ok"}
        if path == "/api/v1/memory/search":
            return {
                "data": {
                    "episodes": [
                        {
                            "id": "ep1",
                            "summary": "You like Yangmei.",
                            "atomic_facts": [
                                {"id": "af1", "content": "You like Yangmei."}
                            ],
                        }
                    ]
                }
            }
        return {"status": "ok"}

    story = cloud.recall_round(
        "我喜欢吃杨梅",
        "我喜欢吃什么",
        base_url="http://server.test",
        session_id="everos-demo-abc",
        user_id="everos_demo_abc",
        request_json=fake_request,
        timeout_seconds=1.0,
    )

    assert [path for _, path, _ in calls] == [
        "/health",
        "/api/v1/memory/add",
        "/api/v1/memory/flush",
        "/api/v1/memory/search",
    ]
    add_body = calls[1][2]
    search_body = calls[3][2]
    assert add_body is not None and search_body is not None
    assert add_body["session_id"] == "everos-demo-abc"
    assert add_body["messages"][0]["sender_id"] == "everos_demo_abc"
    assert add_body["messages"][0]["content"] == "我喜欢吃杨梅"
    assert search_body["user_id"] == "everos_demo_abc"
    assert story.owner == "everos_demo_abc"
    assert story.memory == "我喜欢吃杨梅"
    assert story.query == "我喜欢吃什么"
    assert story.answer == "You like Yangmei."
    assert story.source_filename == "episode:ep1"


def test_recall_round_raises_quota_error_on_429() -> None:
    def fake_request(*_: object, **__: object) -> dict[str, object]:
        raise cloud.CloudQuotaError("http://server.test")

    with pytest.raises(cloud.CloudQuotaError):
        cloud.recall_round(
            "m",
            "q",
            base_url="http://server.test",
            session_id="s",
            user_id="u",
            request_json=fake_request,
        )


def test_recall_round_raises_demo_error_when_search_never_returns() -> None:
    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        if path == "/health":
            return {"status": "ok"}
        if path == "/api/v1/memory/search":
            return {"data": {"episodes": []}}
        return {"status": "ok"}

    with pytest.raises(cloud.CloudDemoError):
        cloud.recall_round(
            "m",
            "q",
            base_url="http://server.test",
            session_id="s",
            user_id="u",
            request_json=fake_request,
            search_attempts=2,
            search_interval_seconds=0.0,
        )
