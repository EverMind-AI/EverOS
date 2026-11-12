"""``content_sha256`` is a required field on every business lancedb table.

Cascade handler (16 doc §3.3) diffs by this digest to skip no-op
re-embeds. Every business schema declares the field; daily-log kinds hash a
per-handler subset of inline + section keys, a profile hashes the file-level
content-bearing parts.
"""

from __future__ import annotations

import datetime as dt

import pytest

from everos.infra.persistence.records import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    UserProfile,
)

_VEC = [0.0] * 1024
_NOW = dt.datetime(2026, 5, 14, 10, 0, 0, tzinfo=dt.UTC)
_SHA = "f" * 64


def _episode() -> Episode:
    return Episode(
        id="u1_ep_1",
        entry_id="ep_20260514_0001",
        owner_id="u1",
        owner_type="user",
        session_id="s1",
        timestamp=_NOW,
        parent_type="memcell",
        parent_id="mc_1",
        sender_ids=["u1"],
        episode="hello world",
        episode_tokens="hello world",
        md_path="users/u1/episodes/episode-2026-05-14.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


def _image_memory() -> ImageMemory:
    return ImageMemory(
        id="u1_img_1",
        entry_id="img_20260514_0001",
        owner_id="u1",
        owner_type="user",
        session_id="s1",
        timestamp=_NOW,
        parent_type="memcell",
        parent_id="mc_1",
        image_id="sha256_abc",
        original_path="figures/abc.png",
        media_type="image/png",
        width=1,
        height=1,
        description="a red square",
        description_tokens="a red square",
        md_path="users/u1/images/image-2026-05-14.md",
        content_sha256=_SHA,
        vector=_VEC,
    )


def _user_profile() -> UserProfile:
    return UserProfile(
        id="u1",
        owner_id="u1",
        owner_type="user",
        summary="likes tea",
        explicit_info_json="[]",
        implicit_traits_json="[]",
        profile_timestamp_ms=1,
        md_path="users/u1/user.md",
        content_sha256=_SHA,
    )


def _group_profile_item() -> GroupProfileItem:
    return GroupProfileItem(
        id="trip_gpi_1",
        entry_id="gpi_20260514_0001",
        owner_id="trip",
        owner_type="group",
        kind="decision",
        statement="we go to Lisbon",
        statement_tokens="we go to Lisbon",
        source_start=_NOW,
        source_end=_NOW,
        observed_at=_NOW,
        payload_json="{}",
        md_path="groups/trip/profile/group_profile-2026-05-14.md",
        content_sha256=_SHA,
    )


_FACTORIES = [_episode, _image_memory, _user_profile, _group_profile_item]
_IDS = ["episode", "image_memory", "user_profile", "group_profile_item"]


@pytest.mark.parametrize("factory", _FACTORIES, ids=_IDS)
def test_content_sha256_round_trip(factory) -> None:  # type: ignore[no-untyped-def]
    row = factory()
    assert row.content_sha256 == _SHA
    dumped = row.model_dump()
    assert dumped["content_sha256"] == _SHA
    restored = type(row).model_validate(dumped)
    assert restored.content_sha256 == _SHA


@pytest.mark.parametrize("factory", _FACTORIES, ids=_IDS)
def test_content_sha256_required(factory) -> None:  # type: ignore[no-untyped-def]
    """Dropping content_sha256 from the kwargs surfaces a ValidationError."""
    row = factory()
    kwargs = row.model_dump()
    del kwargs["content_sha256"]
    with pytest.raises(Exception):  # noqa: B017
        type(row).model_validate(kwargs)
