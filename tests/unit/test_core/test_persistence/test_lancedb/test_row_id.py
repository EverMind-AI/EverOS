"""Storage identity and public wire compatibility contracts."""

from __future__ import annotations

from everos.core.persistence.lancedb.row_id import (
    agent_skill_storage_id,
    agent_skill_wire_id,
    daily_log_storage_id,
    daily_log_wire_id,
    make_storage_id,
    user_profile_storage_id,
    user_profile_wire_id,
)


def test_storage_id_is_injective_across_part_boundaries() -> None:
    assert make_storage_id("a_b", "c") != make_storage_id("a", "b_c")


def test_storage_id_counts_utf8_bytes_and_is_deterministic() -> None:
    assert make_storage_id("é", "x") == "2:é1:x"
    assert make_storage_id("é", "x") == make_storage_id("é", "x")


def test_daily_log_storage_id_changes_with_every_scope_dimension() -> None:
    base = daily_log_storage_id(
        app_id="app_a", project_id="project_a", owner_id="owner", entry_id="ep_1"
    )
    variants = {
        daily_log_storage_id(
            app_id="app_b",
            project_id="project_a",
            owner_id="owner",
            entry_id="ep_1",
        ),
        daily_log_storage_id(
            app_id="app_a",
            project_id="project_b",
            owner_id="owner",
            entry_id="ep_1",
        ),
        daily_log_storage_id(
            app_id="app_a",
            project_id="project_a",
            owner_id="other",
            entry_id="ep_1",
        ),
        daily_log_storage_id(
            app_id="app_a",
            project_id="project_a",
            owner_id="owner",
            entry_id="ep_2",
        ),
    }
    assert base not in variants
    assert len(variants) == 4


def test_kind_specific_storage_ids_include_scope() -> None:
    assert agent_skill_storage_id(
        app_id="app", project_id="one", owner_id="agent", name="skill"
    ) != agent_skill_storage_id(
        app_id="app", project_id="two", owner_id="agent", name="skill"
    )
    assert user_profile_storage_id(
        app_id="app", project_id="one", owner_id="user"
    ) != user_profile_storage_id(app_id="app", project_id="two", owner_id="user")


def test_wire_ids_keep_the_historical_public_shape() -> None:
    assert daily_log_wire_id(owner_id="owner", entry_id="ep_1") == "owner_ep_1"
    assert agent_skill_wire_id(owner_id="agent", name="skill") == "agent_skill"
    assert user_profile_wire_id(owner_id="user") == "user"
