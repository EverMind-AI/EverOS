"""Tests that every business frontmatter class reports the expected
``path_glob()`` — the cascade scanner reads these to enumerate eligible
files, so a wrong glob silently drops a whole kind from cascade.
"""

from __future__ import annotations

import pytest

from everos.infra.persistence.markdown import (
    EpisodeDailyFrontmatter,
    GroupProfileDailyFrontmatter,
    ImageMemoryDailyFrontmatter,
    UserProfileFrontmatter,
)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (EpisodeDailyFrontmatter, "*/*/users/*/episodes/episode-*.md"),
        (ImageMemoryDailyFrontmatter, "*/*/users/*/images/image-*.md"),
        (UserProfileFrontmatter, "*/*/users/*/user.md"),
        (
            GroupProfileDailyFrontmatter,
            "*/*/groups/*/profile/group_profile-*.md",
        ),
    ],
)
def test_path_glob(schema: type, expected: str) -> None:
    assert schema.path_glob() == expected
