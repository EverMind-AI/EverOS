"""Tests that every business frontmatter class reports the expected
``path_glob()`` — the cascade scanner reads these to enumerate eligible
files, so a wrong glob silently drops a whole kind from cascade.
"""

from __future__ import annotations

import pytest

from everos.infra.persistence.markdown import (
    AgentCaseDailyFrontmatter,
    AgentSkillFrontmatter,
    AtomicFactDailyFrontmatter,
    EpisodeDailyFrontmatter,
    ForesightDailyFrontmatter,
)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (EpisodeDailyFrontmatter, "*/*/users/*/episodes/episode-*.md"),
        (AtomicFactDailyFrontmatter, "*/*/users/*/.atomic_facts/atomic_fact-*.md"),
        (ForesightDailyFrontmatter, "*/*/users/*/.foresights/foresight-*.md"),
        (AgentCaseDailyFrontmatter, "*/*/agents/*/.cases/agent_case-*.md"),
        (AgentSkillFrontmatter, "*/*/agents/*/skills/skill_*/SKILL.md"),
    ],
)
def test_path_glob(schema: type, expected: str) -> None:
    assert schema.path_glob() == expected
