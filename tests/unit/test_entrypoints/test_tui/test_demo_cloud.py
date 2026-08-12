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

    assert cloud.resolve_live_base_url(cloud.LIVE_DEMO_SERVER_URL) == (
        cloud.CLOUD_PLATFORM_API_BASE_URL
    )
    assert cloud.resolve_live_base_url("https://live.test") == "https://live.test"


def test_resolve_keys_read_their_env_vars(monkeypatch) -> None:
    monkeypatch.setenv(cloud.CLOUD_DEMO_KEY_ENV, "demo-key")
    monkeypatch.setenv(cloud.CLOUD_USER_KEY_ENV, "user-key")
    assert cloud.resolve_demo_key() == "demo-key"
    assert cloud.resolve_user_key() == "user-key"


def test_public_demo_ships_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv(cloud.CLOUD_DEMO_KEY_ENV, raising=False)
    assert cloud.CLOUD_API_BASE_URL == "https://everosdemo.com"
    assert cloud.resolve_demo_key() == ""


def test_new_demo_identity_is_unique_and_paired() -> None:
    session_a, user_a = cloud.new_demo_identity()
    session_b, user_b = cloud.new_demo_identity()

    assert session_a != session_b
    assert user_a != user_b
    assert session_a.startswith("everos-demo-")
    assert user_a.startswith("everos_demo_")


def test_add_memory_posts_v2_message_synchronously() -> None:
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
        return {"request_id": "req-123", "data": {"message_count": 1}}

    result = cloud.add_memory(
        "我喜欢吃杨梅",
        base_url="https://api.test",
        session_id="everos-demo-abc",
        user_id="everos_demo_abc",
        api_key="k-1",
        request_json=fake_request,
    )

    assert result is None
    method, path, body, api_key = calls[0]
    assert (method, path) == ("POST", "/api/v2/memory/add")
    assert api_key == "k-1"
    assert body["async_mode"] is False
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["sender_id"] == "everos_demo_abc"
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
        assert path == "/api/v2/memory/flush"
        return {"request_id": "req-456", "data": {}}

    cloud.flush_memory(
        base_url="https://api.test",
        session_id="s",
        api_key="k",
        request_json=fake_request,
    )

    assert bodies[0] == {"session_id": "s"}


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
        assert path == "/api/v2/memory/search"
        assert json_body["user_id"] == "everos_demo_abc"
        assert json_body["filters"] == {"session_id": "everos-demo-abc"}
        assert json_body["include_profile"] is True
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
                                "content": "You like Yangmei.",
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
        session_id="everos-demo-abc",
        user_id="everos_demo_abc",
        api_key="k",
        request_json=fake_request,
        settle_seconds=0.0,
    )

    assert story is not None
    assert story.owner == "everos_demo_abc"
    assert story.answer == "You like Yangmei."  # from atomic_fact, not summary
    assert story.score == 0.57  # fact-level score (episode score is null)
    assert story.source_filename == "episode:ep1"


def test_search_recall_picks_highest_scored_episode_not_first() -> None:
    # The platform does not pre-sort episodes by score, so taking episodes[0]
    # blindly can return an unrelated memory. The best-scored one must win.
    def fake_request(*_: object, **kwargs: object) -> dict[str, object]:
        return {
            "data": {
                "episodes": [
                    {"id": "ep_durian", "summary": "dislikes durian", "score": None},
                    {
                        "id": "ep_climb",
                        "summary": "climbs in Yosemite",
                        "score": 0.61,
                    },
                ]
            }
        }

    story = cloud.search_recall(
        "I climb in Yosemite",
        "Where do I climb?",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        settle_seconds=0.0,
    )

    assert story is not None
    assert story.source_filename == "episode:ep_climb"  # not ep_durian at [0]
    assert story.score == 0.61


def test_search_recall_prefers_higher_scored_profile_over_episode() -> None:
    # Profiles are concise, answer-shaped facts that score well on natural
    # questions; when a profile out-scores the episodes, it must win.
    def fake_request(*_: object, **__: object) -> dict[str, object]:
        return {
            "data": {
                "profiles": [
                    {
                        "id": "pr1",
                        "score": 0.72,
                        "profile_data": {
                            "embed_text": "hobby: Enjoys climbing in Yosemite"
                        },
                    }
                ],
                "episodes": [
                    {"id": "ep_durian", "summary": "dislikes durian", "score": 0.40},
                ],
            }
        }

    story = cloud.search_recall(
        "I climb in Yosemite",
        "Where do I climb?",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        settle_seconds=0.0,
    )

    assert story is not None
    assert story.answer == "Enjoys climbing in Yosemite"  # profile, "hobby:" dropped
    assert story.score == 0.72
    assert story.source_filename == "profile:pr1"


def test_search_recall_waits_for_current_memory_instead_of_stale_profile() -> None:
    calls = 0

    def fake_request(*_: object, **__: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        stale_profile = {
            "id": "old-food",
            "score": 0.68,
            "profile_data": {"embed_text": "food preference: 我爱吃草莓"},
        }
        if calls == 1:
            return {"data": {"profiles": [stale_profile], "episodes": []}}
        return {
            "data": {
                "profiles": [stale_profile],
                "episodes": [
                    {
                        "id": "current-durian",
                        "score": 0.56,
                        "atomic_facts": [
                            {
                                "id": "fact-durian",
                                "atomic_fact": "我不爱吃榴莲",
                            }
                        ],
                    }
                ],
            }
        }

    story = cloud.search_recall(
        "我不爱吃榴莲",
        "我不爱吃什么",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        search_attempts=2,
        search_interval_seconds=0.0,
        settle_seconds=0.0,
    )

    assert calls == 2
    assert story is not None
    assert story.answer == "我不爱吃榴莲"
    assert story.source_filename == "episode:current-durian"


def test_current_memory_match_penalizes_opposite_preference() -> None:
    memory = "我不爱吃榴莲"

    current = cloud._memory_match_score(memory, "用户不喜欢吃榴莲")
    stale = cloud._memory_match_score(memory, "我爱吃草莓")

    assert current >= cloud.CURRENT_MEMORY_MATCH_THRESHOLD
    assert current > stale


def test_search_recall_below_relevance_floor_is_a_miss() -> None:
    # An off-topic query still gets a best-but-weak candidate from the platform;
    # below the relevance floor we must report a miss, not an absurd answer.
    def fake_request(*_: object, **__: object) -> dict[str, object]:
        return {
            "data": {
                "profiles": [
                    {
                        "id": "p",
                        "score": 0.40,
                        "profile_data": {
                            "embed_text": "food preference: 用户喜欢吃杨梅"
                        },
                    }
                ],
                # v2 may also expose the just-written message while extraction
                # is settling; an unrelated question must not surface it.
                "unprocessed_messages": [{"id": "buffered", "content": "我喜欢吃杨梅"}],
            }
        }

    story = cloud.search_recall(
        "我喜欢吃杨梅",
        "我是程序员吗",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        settle_seconds=0.0,
        search_interval_seconds=0.0,
    )

    assert story is None  # 0.40 < MIN_RELEVANCE_SCORE


def test_search_recall_accepts_low_score_when_it_is_clearly_this_round() -> None:
    calls = 0

    def fake_request(*_: object, **__: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "data": {
                "episodes": [
                    {
                        "id": "current-strawberry",
                        "score": 0.49,
                        "atomic_facts": [
                            {
                                "id": "fact-strawberry",
                                "content": "我喜欢吃草莓",
                                "score": 0.49,
                            }
                        ],
                    }
                ]
            }
        }

    story = cloud.search_recall(
        "我喜欢吃草莓",
        "我喜欢吃什么",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        settle_seconds=0.0,
        search_interval_seconds=0.0,
    )

    assert calls == 1
    assert story is not None
    assert story.answer == "我喜欢吃草莓"
    assert story.score == 0.49


def test_search_recall_retries_translated_low_score_then_uses_current_memory() -> None:
    """A translated v2 candidate must not stop polling and become a false miss."""

    calls = 0

    def fake_request(*_: object, **__: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "data": {
                "episodes": [
                    {
                        "id": "current-strawberry",
                        "score": 0.49,
                        "atomic_facts": [
                            {
                                "id": "fact-strawberry",
                                "content": "The user likes strawberries.",
                                "score": 0.49,
                            }
                        ],
                    }
                ]
            }
        }

    story = cloud.search_recall(
        "我喜欢吃草莓",
        "我喜欢吃什么",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        search_attempts=2,
        settle_seconds=0.0,
        search_interval_seconds=0.0,
    )

    assert calls == 2
    assert story is not None
    assert story.answer == "我喜欢吃草莓"
    assert story.source_filename == "buffer:current"
    assert story.score == 0.0


def test_search_recall_can_fall_back_to_an_earlier_demo_round() -> None:
    """A question may target any memory stored earlier in the same demo run."""

    def fake_request(*_: object, **__: object) -> dict[str, object]:
        return {
            "data": {
                "episodes": [
                    {
                        "id": "strawberry",
                        "score": 0.49,
                        "atomic_facts": [
                            {
                                "id": "fact-strawberry",
                                "content": "The user likes strawberries.",
                                "score": 0.49,
                            }
                        ],
                    }
                ]
            }
        }

    story = cloud.search_recall(
        "我不喜欢吃榴莲",
        "我喜欢吃什么",
        stored_memories=["我喜欢吃草莓", "我不喜欢吃榴莲"],
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        search_attempts=2,
        settle_seconds=0.0,
        search_interval_seconds=0.0,
    )

    assert story is not None
    assert story.answer == "我喜欢吃草莓"
    assert story.source_filename == "buffer:history"
    assert story.score == 0.0


def test_search_recall_uses_v2_unprocessed_message_after_polling() -> None:
    calls = 0

    def fake_request(*_: object, **__: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "data": {
                "episodes": [],
                "profiles": [],
                "unprocessed_messages": [
                    {
                        "id": "message-strawberry",
                        "session_id": "s",
                        "sender_id": "u",
                        "content": "我喜欢吃草莓",
                    }
                ],
            }
        }

    story = cloud.search_recall(
        "我喜欢吃草莓",
        "我喜欢吃什么",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        search_attempts=2,
        settle_seconds=0.0,
        search_interval_seconds=0.0,
    )

    assert calls == 2
    assert story is not None
    assert story.answer == "我喜欢吃草莓"
    assert story.source_filename == "buffer:message-stra"
    assert story.score == 0.0


def test_clean_profile_text_strips_label_across_colon_widths() -> None:
    assert cloud._clean_profile_text("hobby: Enjoys climbing") == "Enjoys climbing"
    zh = cloud._clean_profile_text("food preference: 用户喜欢吃杨梅")
    assert zh == "用户喜欢吃杨梅"
    assert cloud._clean_profile_text("爱好：喜欢爬山") == "喜欢爬山"
    # No short label -> keep the whole text.
    assert cloud._clean_profile_text("just a plain sentence") == "just a plain sentence"


def test_search_recall_returns_none_on_miss() -> None:
    def fake_request(*_: object, **__: object) -> dict[str, object]:
        return {"data": {"episodes": []}}

    story = cloud.search_recall(
        "m",
        "q",
        base_url="https://api.test",
        session_id="s",
        user_id="u",
        api_key="k",
        request_json=fake_request,
        search_attempts=2,
        search_interval_seconds=0.0,
        settle_seconds=0.0,
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
