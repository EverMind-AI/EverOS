"""The bootstrap's profile choices: who gets the cue block, and when a profile updates.

What these pin:

- the episode call carries the ``<profile_cues>`` block only with
  ``[profile] input = "cues"``: for a user's own space while
  ``extract_user_profile`` runs, and for a group's while
  ``update_member_profiles`` runs in ``[groups] member_profile_mode =
  "batched"`` (``per_member`` reads every message); an ``ome.toml`` that
  turns either off at startup turns its block off, a file that cannot be
  read changes nothing, and a server without its engine asks for no cue;
- ``build_application`` hands the conversation pipeline that decision and the
  strategies the ``[profile]`` input, and the memory service built without
  its engine asks for none;
- a user's profile updates at a flush only with ``[profile]
  update_on_flush``, on its own idle trigger after ``[profile]
  idle_seconds``, and on a catch-up's ``UserProfileDue`` either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.config import Settings, load_settings
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, SessionFlushed, UserProfileDue
from everos.memory.extract.pipeline import CueConsumers
from everos.memory.ports import Idle, Immediate, StrategySpec
from everos.memory.strategies import (
    EXTRACT_USER_PROFILE_IDLE_SPEC,
    EXTRACT_USER_PROFILE_SPEC,
)
from everos.service import bootstrap, build_application
from everos.service.bootstrap import configured_spec, profile_cue_consumers
from tests.fakes import make_providers


def _profile(**values: object) -> Settings:
    base = Settings()
    return base.model_copy(update={"profile": base.profile.model_copy(update=values)})


def test_by_default_both_spaces_read_cues(tmp_path: Path) -> None:
    assert profile_cue_consumers(Settings(), tmp_path / "ome.toml") == CueConsumers(
        users=True, groups=True
    )


def test_a_full_profile_reads_no_cue_in_either_space(tmp_path: Path) -> None:
    consumers = profile_cue_consumers(_profile(input="full"), tmp_path / "ome.toml")
    assert consumers == CueConsumers(users=False, groups=False)


def test_per_member_profiles_read_no_cue(tmp_path: Path) -> None:
    base = Settings()
    per_member = base.model_copy(
        update={
            "groups": base.groups.model_copy(
                update={"member_profile_mode": "per_member"}
            )
        }
    )
    consumers = profile_cue_consumers(per_member, tmp_path / "ome.toml")
    assert consumers == CueConsumers(users=True, groups=False)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("extract_user_profile", CueConsumers(users=False, groups=True)),
        ("update_member_profiles", CueConsumers(users=True, groups=False)),
    ],
)
def test_a_profile_ome_toml_turns_off_gets_no_block(
    tmp_path: Path, section: str, expected: CueConsumers
) -> None:
    path = tmp_path / "ome.toml"
    path.write_text(f"[strategies.{section}]\nenabled = false\n", encoding="utf-8")
    assert profile_cue_consumers(Settings(), path) == expected


def test_an_unreadable_ome_toml_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "ome.toml"
    path.write_text("[strategies.extract_user_profile\nenabled = ", encoding="utf-8")
    assert profile_cue_consumers(Settings(), path) == CueConsumers(
        users=True, groups=True
    )


def test_a_server_without_its_engine_asks_for_no_cue(tmp_path: Path) -> None:
    base = Settings()
    off = base.model_copy(
        update={"ome": base.ome.model_copy(update={"enabled": False})}
    )
    assert profile_cue_consumers(off, tmp_path / "ome.toml") == CueConsumers()


def test_the_application_hands_the_pipeline_and_the_strategies_their_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("EVEROS_PROFILE__INPUT", "full")
    monkeypatch.setenv("EVEROS_PROFILE__PICTURE_PIXELS", "true")
    bound: list[MemoryDeps] = []
    original = bootstrap.to_ome_strategy

    def binding(spec: StrategySpec, deps: MemoryDeps) -> object:
        bound.append(deps)
        return original(spec, deps)

    monkeypatch.setattr(bootstrap, "to_ome_strategy", binding)
    load_settings.cache_clear()
    try:
        application = build_application(load_settings(), providers=make_providers())
    finally:
        load_settings.cache_clear()

    pipeline = application.memory._memorizer._conversation_pipeline
    assert pipeline._cue_consumers == CueConsumers()  # type: ignore[attr-defined]
    assert pipeline._profile_cues is not None  # type: ignore[attr-defined]
    assert {(d.profile_input, d.profile_picture_pixels) for d in bound} == {
        ("full", True)
    }
    assert all(d.stores.profile_cues is not None for d in bound)


@pytest.mark.parametrize("on_flush", [False, True])
def test_a_flush_updates_a_user_s_profile_only_when_configured(on_flush: bool) -> None:
    spec = configured_spec(
        EXTRACT_USER_PROFILE_SPEC, _profile(update_on_flush=on_flush)
    )
    assert isinstance(spec.trigger, Immediate)
    expected = (
        [EpisodeExtracted, SessionFlushed, UserProfileDue]
        if on_flush
        else [EpisodeExtracted, UserProfileDue]
    )
    assert list(spec.trigger.on) == expected


@pytest.mark.parametrize(("seconds", "scan"), [(600, 60), (40, 20)])
def test_the_user_profile_s_idle_trigger_takes_the_configured_seconds(
    seconds: int, scan: int
) -> None:
    spec = configured_spec(
        EXTRACT_USER_PROFILE_IDLE_SPEC, _profile(idle_seconds=seconds)
    )
    assert isinstance(spec.trigger, Idle)
    assert (spec.trigger.idle_seconds, spec.trigger.scan_interval_seconds) == (
        seconds,
        scan,
    )
    assert spec.trigger.event_field == "user_key"
