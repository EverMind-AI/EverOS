"""Tests for the cascade kind registry.

Verify the registered kinds' globs match the right paths and reject
noise (random ``.md``, swp files, profile-style paths, the group profile's
``pending.md``, and the atomic-fact / foresight logs a store written before
EverOS 2.0 still holds). ``match_kind`` must walk the registry in declared
order and pick the first matching spec.
"""

from __future__ import annotations

import pytest

from everos.infra.cascade import KIND_REGISTRY, match_kind


@pytest.mark.parametrize(
    ("path", "expected_kind"),
    [
        (
            "default_app/default_project/users/u1/episodes/episode-2026-05-14.md",
            "episode",
        ),
        ("claude_code/oss/users/u_jason/episodes/episode-2026-01-01.md", "episode"),
        (
            "default_app/default_project/agents/a1/.cases/agent_case-2026-05-14.md",
            "agent_case",
        ),
        (
            "default_app/default_project/agents/a1/skills/skill_contract_risk_scan/SKILL.md",
            "agent_skill",
        ),
        (
            "default_app/p/groups/trip/profile/group_profile-2026-03-02.md",
            "group_profile",
        ),
        (
            "default_app/default_project/users/u1/scenes/scene-2026-05-14.md",
            "scene",
        ),
        ("default_app/p/groups/trip/scenes/scene-2026-03-02.md", "group_scene"),
    ],
)
def test_match_kind_recognises_registered_paths(path: str, expected_kind: str) -> None:
    spec = match_kind(path)
    assert spec is not None
    assert spec.name == expected_kind


@pytest.mark.parametrize(
    "path",
    [
        "users/u1/profile/user.md",
        "users/u1/random.md",
        "users/u1/episodes/draft.txt",  # wrong extension
        ".cache/foo.md",
        "users/u1/episodes/episode-2026-05-14.md.swp",  # swap file
        "agents/a1/skills/skill_x/references/notes.md",  # reference, not main
        # Valid episode shape but MISSING the <app>/<project> prefix — must be
        # rejected so a prefix-less path can never silently match (the scanner
        # would otherwise find nothing while the watcher matched, a split brain).
        "users/u1/episodes/episode-2026-05-14.md",
        # The group profile's pending batch is state beside the logs, never
        # indexed.
        "default_app/p/groups/trip/profile/pending.md",
        # Logs of the kinds EverOS 2.0 removed: an older store still has them
        # on disk, and the cascade must leave them alone.
        "default_app/default_project/users/u1/.atomic_facts/atomic_fact-2026-05-14.md",
        "default_app/default_project/users/u1/.foresights/foresight-2026-05-14.md",
    ],
)
def test_match_kind_rejects_unregistered_paths(path: str) -> None:
    assert match_kind(path) is None


def test_registry_has_exactly_fourteen_kinds() -> None:
    """The registry pins the cascade surface — no silent registration."""
    names = [s.name for s in KIND_REGISTRY]
    assert names == [
        "episode",
        "agent_case",
        "agent_skill",
        "image_memory",
        "user_profile",
        "scene",
        "group_episode",
        "group_image_memory",
        "group_scene",
        "group_member_profile",
        "group_profile",
        "group_roster",
        "knowledge_document",
        "knowledge_topic",
    ]


def test_kind_spec_path_glob_reads_off_schema() -> None:
    """Path glob is owned by the frontmatter schema, not duplicated here."""
    for spec in KIND_REGISTRY:
        assert spec.path_glob() == spec.frontmatter_schema.path_glob()
