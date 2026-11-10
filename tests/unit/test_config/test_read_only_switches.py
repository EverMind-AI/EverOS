"""The evaluation harness's read-only switches reach the settings they stand for.

The harness that produced the published numbers starts a search-only server with
``EVEROS_DISABLE_CASCADE=1`` and ``EVEROS_DISABLE_OME=1``, and every server with
``EVEROS_DISABLE_CASCADE_WATCHER=1``. The exp tree honours all three; this tree had
renamed the first two to ``[cascade] enabled`` / ``watcher_enabled`` and had no OME
switch, so the same launch silently ran both write-side subsystems on a read-only
server. Each switch now clears its setting -- and only when set to a value the exp
tree reads as on. The default is the part pinned hardest: a switch that read as on
by accident would silently stop extraction.
"""

from __future__ import annotations

import pytest

from everos.config import load_settings
from everos.config.settings import CascadeSettings, OmeSettings, Settings

_SWITCHES = {
    "EVEROS_DISABLE_CASCADE": ("cascade", "enabled"),
    "EVEROS_DISABLE_CASCADE_WATCHER": ("cascade", "watcher_enabled"),
    "EVEROS_DISABLE_OME": ("ome", "enabled"),
}
TRUTHY = ["1", "true", "TRUE", "yes", "Yes", " 1 "]
FALSY = ["", "   ", "0", "false", "no", "off", "maybe"]


def _value(settings: Settings, section: str, key: str) -> bool:
    return getattr(getattr(settings, section), key)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in (*_SWITCHES, "EVEROS_CASCADE__ENABLED", "EVEROS_OME__ENABLED"):
        monkeypatch.delenv(env, raising=False)
    load_settings.cache_clear()


@pytest.mark.parametrize("env", list(_SWITCHES))
@pytest.mark.parametrize("value", TRUTHY)
def test_a_switch_that_is_on_clears_its_setting(
    env: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, value)
    section, key = _SWITCHES[env]
    assert _value(Settings(), section, key) is False


@pytest.mark.parametrize("env", list(_SWITCHES))
@pytest.mark.parametrize("value", FALSY)
def test_anything_else_leaves_the_setting_alone(
    env: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, value)
    section, key = _SWITCHES[env]
    assert _value(Settings(), section, key) is True


def test_unset_leaves_everything_on() -> None:
    """The default. A run that ingests must not have to know these exist."""
    settings = Settings()
    assert (settings.cascade.enabled, settings.cascade.watcher_enabled) == (True, True)
    assert settings.ome.enabled is True


def test_each_switch_clears_only_its_own_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watcher switch keeps the worker, so md still reaches the index."""
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE_WATCHER", "1")
    settings = Settings()
    assert settings.cascade.watcher_enabled is False
    assert settings.cascade.enabled is True
    assert settings.ome.enabled is True


def test_the_settings_own_names_keep_working(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_CASCADE__ENABLED", "false")
    monkeypatch.setenv("EVEROS_OME__ENABLED", "false")
    settings = Settings()
    assert (settings.cascade.enabled, settings.ome.enabled) == (False, False)


def test_a_switch_wins_over_the_settings_own_env_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As the legacy ``EVEROS_LLMMR_*`` names do; the section's other keys merge."""
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE", "1")
    monkeypatch.setenv("EVEROS_CASCADE__ENABLED", "true")
    monkeypatch.setenv("EVEROS_CASCADE__SCAN_INTERVAL_SECONDS", "7")
    settings = Settings()
    assert settings.cascade.enabled is False
    assert settings.cascade.scan_interval_seconds == 7.0


def test_init_arguments_still_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE", "1")
    monkeypatch.setenv("EVEROS_DISABLE_OME", "1")
    settings = Settings(
        cascade=CascadeSettings(enabled=True), ome=OmeSettings(enabled=True)
    )
    assert (settings.cascade.enabled, settings.ome.enabled) == (True, True)


def test_load_settings_sees_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached accessor the bootstrap reads goes through the same sources."""
    monkeypatch.setenv("EVEROS_DISABLE_OME", "yes")
    load_settings.cache_clear()
    assert load_settings().ome.enabled is False
