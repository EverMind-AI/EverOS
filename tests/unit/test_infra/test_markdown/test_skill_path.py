"""Tests for semantic skill name to filesystem-component encoding."""

from __future__ import annotations

import pytest

from everos.infra.persistence.markdown.skill_path import (
    encode_skill_name_segment,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("alpha", "alpha"),
        ("API verification", "API verification"),
        ("平台实例判定", "平台实例判定"),
        ("框架/平台/实例", "框架%2F平台%2F实例"),
        (r"SDK\API", "SDK%5CAPI"),
        ("literal%2Fname", "literal%252Fname"),
        (r"50%/SDK\API", "50%25%2FSDK%5CAPI"),
    ],
)
def test_encode_skill_name_segment(name: str, expected: str) -> None:
    assert encode_skill_name_segment(name) == expected


def test_encoding_is_collision_safe_for_literal_escape_text() -> None:
    assert encode_skill_name_segment("a/b") == "a%2Fb"
    assert encode_skill_name_segment("a%2Fb") == "a%252Fb"
    assert encode_skill_name_segment("a/b") != encode_skill_name_segment("a%2Fb")
