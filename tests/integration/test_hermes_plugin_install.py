"""Integration smoke: ``everos integrations install`` + plugin load.

Runs the real ``install`` command (Typer CliRunner) against the actual
``integrations/hermes/`` bundle and a throwaway ``HERMES_HOME``, then
imports the symlinked bundle with Hermes stubs injected and asserts the
``EverosMemoryProvider`` is loadable as a ``MemoryProvider`` subclass.

Deviation from the written spec: the bundle's ``__init__.py`` does not
define a ``register(ctx)`` function — Hermes' plugin loader
(``plugins/memory/__init__.py:_load_provider_from_dir``) falls back to
instantiate a ``MemoryProvider`` subclass found on the module. This test
pins that fallback path (the actual contract) rather than the absent
``register`` entry point.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import integrations as integrations_mod

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_BUNDLE = _REPO_ROOT / "integrations" / "hermes"


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("EVEROS_HERMES_PLUGIN_SOURCE", raising=False)
    return home


def _install_symlink(hermes_home: Path) -> Path:
    """Run the real install command against the real bundle; return target."""
    result = CliRunner().invoke(
        integrations_mod.app,
        ["install", "hermes", "--source", str(_REAL_BUNDLE)],
    )
    assert result.exit_code == 0, result.output
    target = hermes_home / "plugins" / "everos"
    assert target.is_symlink(), f"expected symlink at {target}"
    assert target.resolve() == _REAL_BUNDLE.resolve()
    return target


def _inject_hermes_stubs() -> None:
    """Install the Hermes-runtime stubs into ``sys.modules``.

    Idempotent: existing entries are left in place so re-injection does not
    clobber a previously-imported plugin module.
    """
    import tests.helpers.hermes_stub as stub

    mods = {
        "agent": types.ModuleType("agent"),
        "agent.memory_provider": stub,
        "hermes_constants": stub,
        "tools": types.ModuleType("tools"),
        "tools.registry": stub,
        "utils": stub,
    }
    mods["agent"].memory_provider = stub  # type: ignore[attr-defined]
    mods["tools"].registry = stub  # type: ignore[attr-defined]
    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def test_install_creates_symlink_pointing_at_bundle(hermes_home: Path):
    target = _install_symlink(hermes_home)
    assert target.is_symlink()
    assert target.resolve() == _REAL_BUNDLE.resolve()
    # The bundle's plugin manifest is reachable through the symlink.
    assert (target / "plugin.yaml").is_file()
    assert (target / "__init__.py").is_file()


def test_installed_plugin_loads_as_memory_provider(hermes_home: Path):
    _install_symlink(hermes_home)
    _inject_hermes_stubs()

    # Import the bundle as a package. The symlink makes the real bundle
    # reachable through HERMES_HOME too, but importing by the canonical
    # name exercises the same __init__.py the Hermes loader would exec.
    plugin = importlib.import_module("integrations.hermes")

    # The plugin does not expose register(ctx); Hermes' loader falls back to
    # finding a MemoryProvider subclass. Pin that contract.
    assert not hasattr(plugin, "register"), (
        "bundle now defines register(); update this test to the new path"
    )
    from tests.helpers.hermes_stub import MemoryProvider

    assert hasattr(plugin, "EverosMemoryProvider")
    provider_cls = plugin.EverosMemoryProvider
    assert isinstance(provider_cls, type)
    assert issubclass(provider_cls, MemoryProvider)
    # The provider is instantiable (the loader's fallback does this).
    provider = provider_cls()
    assert provider.name == "everos"


def test_installed_plugin_handles_tool_call_end_to_end(hermes_home: Path):
    _install_symlink(hermes_home)
    _inject_hermes_stubs()

    plugin = importlib.import_module("integrations.hermes")
    from tests.helpers.hermes_stub import MemoryProvider

    provider = plugin.EverosMemoryProvider()
    assert isinstance(provider, MemoryProvider)
    # The four tool schemas are exposed regardless of backend availability.
    schemas = {s["name"] for s in provider.get_tool_schemas()}
    assert schemas == {"everos_search", "everos_list", "everos_add", "everos_flush"}
    # Without a configured client, tool calls surface a clean error.
    provider._client = None
    provider._init_error = "not configured"
    out = provider.handle_tool_call("everos_search", {"query": "x"})
    assert "error" in json.loads(out)


def test_uninstall_after_install_round_trip(hermes_home: Path):
    target = _install_symlink(hermes_home)
    result = CliRunner().invoke(
        integrations_mod.app,
        ["uninstall", "hermes", "--source", str(_REAL_BUNDLE)],
    )
    assert result.exit_code == 0, result.output
    assert not target.exists()
