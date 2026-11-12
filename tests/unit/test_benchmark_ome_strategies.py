"""The benchmark turns OME strategies off through ``ome.toml``, the file EverOS reads.

LoCoMo answers without the user profile, yet every server ran the profile strategy,
spending calls and, when a run failed for good, stopping the conversation over output
nothing reads. Strategy switches have no environment variable, so the config's
``[ome_strategies]`` is written as each started server's ``ome.toml``.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from everos.infra.ome.config import TomlRoot

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")
BenchmarkConfig = importlib.import_module("config").BenchmarkConfig


def test_the_written_file_is_one_ome_reads(tmp_path: Path) -> None:
    run.write_ome_strategies(
        tmp_path,
        {
            "extract_user_profile": {"enabled": False},
            "extract_agent_skill": {"max_retries": 1, "gate": {"threshold": 3}},
        },
    )
    parsed = TomlRoot.model_validate(
        tomllib.loads((tmp_path / "ome.toml").read_text(encoding="utf-8"))
    )
    assert parsed.strategies["extract_user_profile"].enabled is False
    skill = parsed.strategies["extract_agent_skill"]
    assert skill.max_retries == 1
    assert skill.gate is not None and skill.gate.threshold == 3


def test_a_rerun_replaces_the_file(tmp_path: Path) -> None:
    run.write_ome_strategies(tmp_path, {"extract_user_profile": {"enabled": False}})
    run.write_ome_strategies(tmp_path, {"extract_foresight": {"enabled": True}})
    parsed = tomllib.loads((tmp_path / "ome.toml").read_text(encoding="utf-8"))
    assert parsed == {"strategies": {"extract_foresight": {"enabled": True}}}


def test_the_fleet_writes_it_before_a_server_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    class _Proc:
        def __init__(self, *_a: Any, **kw: Any) -> None:
            seen.append((Path(kw["env"]["EVEROS_ROOT"]) / "ome.toml").read_text())

        def poll(self) -> None:
            return None

    class _NoWaitFleet(run._ServerFleet):
        def _await_ready(self, timeout_s: float = 180.0) -> None:
            return None

    monkeypatch.setattr(run.subprocess, "Popen", _Proc)
    root = tmp_path / "store"
    root.mkdir()
    (root / "everos.toml").write_text("", encoding="utf-8")
    off = {"extract_user_profile": {"enabled": False}}
    _NoWaitFleet(1, root, first_port=0, ome_strategies=off).start()
    (text,) = seen
    assert tomllib.loads(text) == {"strategies": off}


def test_locomo_runs_without_the_profile() -> None:
    config = BenchmarkConfig.from_toml("locomo")
    assert config.include_profile is False
    assert config.ome_strategies == {"extract_user_profile": {"enabled": False}}
