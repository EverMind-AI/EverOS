"""Two ways the CLI failed on inputs and platforms it documents as supported.

Both are the same shape: an assumption that held on the machine the harness was
written on, applied unconditionally.

* ``--config`` is documented as pointing at a file outside ``benchmarks/configs/``,
  and ``from_toml`` appended ``.toml`` to whatever it was handed -- so a full path
  came back as ``FileNotFoundError`` naming ``custom.toml.toml``, a file nobody asked
  for.
* ``--list-servers`` walks ``/proc``, which only Linux has, and did so without
  checking, so the command ended in a traceback anywhere else.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from config import BenchmarkConfig  # noqa: E402

run = importlib.import_module("run")

_BUILTIN = _BENCH / "configs"


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('adapter = "locomo"\ntop_k = 7\n', encoding="utf-8")
    return path


def test_an_absolute_path_is_opened_as_given(tmp_path: Path) -> None:
    """The reported case: ``--config /tmp/custom.toml``."""
    cfg = BenchmarkConfig.from_toml(str(_write(tmp_path / "custom.toml")))
    assert cfg.top_k == 7


def test_a_relative_path_is_opened_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "sub" / "custom.toml")
    monkeypatch.chdir(tmp_path)
    assert BenchmarkConfig.from_toml("sub/custom.toml").top_k == 7


def test_a_path_without_the_suffix_is_still_a_path(tmp_path: Path) -> None:
    """A separator is enough to say "this is a file", so nothing is appended."""
    target = tmp_path / "sub" / "custom"
    _write(target)
    assert BenchmarkConfig.from_toml(str(target)).top_k == 7


@pytest.mark.parametrize("name", ["locomo", "config.locomo"])
def test_a_built_in_name_still_resolves_under_configs(name: str) -> None:
    """The common path, and the legacy ``config.<name>`` spelling, are unchanged."""
    assert BenchmarkConfig.from_toml(name).adapter == "locomo"


def test_the_error_names_the_path_it_actually_opened(tmp_path: Path) -> None:
    """Otherwise a mis-resolved name is indistinguishable from a missing file."""
    missing = tmp_path / "absent.toml"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        BenchmarkConfig.from_toml(str(missing))

    with pytest.raises(FileNotFoundError, match=r"nosuchbenchmark\.toml"):
        BenchmarkConfig.from_toml("nosuchbenchmark")


def test_a_suffixed_name_is_never_suffixed_twice(tmp_path: Path) -> None:
    """The defect itself, stated directly."""
    with pytest.raises(FileNotFoundError) as err:
        BenchmarkConfig.from_toml(str(tmp_path / "custom.toml"))
    assert ".toml.toml" not in str(err.value)


# ---------------------------------------------------------------------------
# --list-servers off Linux
# ---------------------------------------------------------------------------


def test_no_proc_reports_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """macOS has no ``/proc``; the command must end, not raise."""
    real_is_dir = Path.is_dir

    def _fake_is_dir(self: Path) -> bool:
        return False if str(self) == "/proc" else real_is_dir(self)

    def _boom(self: Path) -> Any:
        raise AssertionError("must not walk /proc when it is not there")

    monkeypatch.setattr(Path, "is_dir", _fake_is_dir)
    monkeypatch.setattr(Path, "iterdir", _boom)

    assert run._iter_servers() == []
    out = capsys.readouterr().out
    assert "/proc" in out


def test_linux_still_reads_proc() -> None:
    """The behaviour that exists is not traded away for the guard.

    Asserted by the call completing on this Linux runner: a guard that swallowed
    every platform would return the same empty list and look identical.
    """
    assert isinstance(run._iter_servers(), list)
    assert Path("/proc").is_dir(), "this test is only meaningful where /proc exists"
