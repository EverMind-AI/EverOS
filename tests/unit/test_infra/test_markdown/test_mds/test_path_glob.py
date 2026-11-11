"""Tests that every business frontmatter class reports the expected
``path_glob()`` — the cascade scanner reads these to enumerate eligible
files, so a wrong glob silently drops a whole kind from cascade.
"""

from __future__ import annotations

import pytest

from everos.infra.persistence.markdown import (
    AgentCaseDailyFrontmatter,
    AgentSkillFrontmatter,
    EpisodeDailyFrontmatter,
    GroupProfileDailyFrontmatter,
)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (EpisodeDailyFrontmatter, "*/*/users/*/episodes/episode-*.md"),
        (AgentCaseDailyFrontmatter, "*/*/agents/*/.cases/agent_case-*.md"),
        (AgentSkillFrontmatter, "*/*/agents/*/skills/skill_*/SKILL.md"),
        (
            GroupProfileDailyFrontmatter,
            "*/*/groups/*/profile/group_profile-*.md",
        ),
    ],
)
def test_path_glob(schema: type, expected: str) -> None:
    assert schema.path_glob() == expected
