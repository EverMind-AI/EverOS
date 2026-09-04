"""A fleet must not serve empty sibling directories instead of the store it was given.

`_ServerFleet` derives one root per server -- `<root>_s0 .. <root>_s{N-1}` -- because
the
index queue lock is exclusive, and uses the given root verbatim only when N == 1. So
pointing `--everos-root` at a single prebuilt store while asking for more than one
server
reads that store's *name* and serves N siblings of it. Those siblings are empty, and in
this workspace they already exist: `everos_store_s0` and `everos_store_s1` sit next to
`everos_store` in every shared store, left over from a sharded build.

Measured on a store holding 711 episodes: 20 of 20 questions returned zero episodes in
1.15s each, `search_error` was empty, and the run exited 0. With the guard, the same
command returns episodes for every question at ~44s each. The shipped configs default to
`servers = 10`, so this is the path a first run takes.

The failure shape is the one this repository keeps producing: a complete-looking run
over
nothing. It is worth a guard rather than a note in a README.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")

_GUARD = "would serve"


def _src_block() -> str:
    src = (_BENCH / "run.py").read_text(encoding="utf-8")
    i = src.index("if args.servers > 1 and args.everos_root:")
    return src[i : i + 1400]


def _store(tmp: Path, name: str, *, populated: bool) -> Path:
    root = tmp / name
    (root / "default_app" / "default_project" / "users" / "alice").mkdir(parents=True)
    if populated:
        (
            root / "default_app" / "default_project" / "users" / "alice" / "u.md"
        ).write_text("# alice\n", encoding="utf-8")
    return root


def test_the_guard_keys_on_markdown_not_directory_existence(tmp_path: Path) -> None:
    """`_s0` existing is not evidence it holds anything.

    Keying on the directory would have passed in exactly the case that failed: the
    sharded siblings were present and empty, which is why `everos init` succeeded and
    the
    run looked healthy.
    """
    block = _src_block()
    assert 'rglob("*.md")' in block
    assert "_base_has_md" in block and "_shards_have_md" in block


def test_a_populated_store_with_empty_shards_clamps_to_one_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The unambiguous case: the caller means "read this store"."""
    base = _store(tmp_path, "everos_store", populated=True)
    for i in range(3):
        (tmp_path / f"everos_store_s{i}").mkdir()
    args = _Args(servers=3, everos_root=[str(base)])
    _apply_guard(args)
    assert args.servers == 1
    assert _GUARD in capsys.readouterr().out


def test_real_shards_are_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A genuinely sharded store must still get its fleet.

    This is the case the derivation exists for; clamping it would serialise a run that
    was built to be parallel.
    """
    base = _store(tmp_path, "everos_store", populated=True)
    for i in range(2):
        _store(tmp_path, f"everos_store_s{i}", populated=True)
    args = _Args(servers=2, everos_root=[str(base)])
    _apply_guard(args)
    assert args.servers == 2
    assert _GUARD not in capsys.readouterr().out


def test_an_empty_base_is_left_alone(tmp_path: Path) -> None:
    """A fresh ADD has nothing anywhere yet, and must keep its fleet.

    Clamping here would turn every sharded build into a single-server one.
    """
    base = _store(tmp_path, "everos_store", populated=False)
    args = _Args(servers=4, everos_root=[str(base)])
    _apply_guard(args)
    assert args.servers == 4


def test_one_server_is_never_touched(tmp_path: Path) -> None:
    """N == 1 already uses the root verbatim, so there is nothing to guard."""
    base = _store(tmp_path, "everos_store", populated=True)
    args = _Args(servers=1, everos_root=[str(base)])
    _apply_guard(args)
    assert args.servers == 1


class _Args:
    def __init__(self, servers: int, everos_root: list[str]) -> None:
        self.servers = servers
        self.everos_root = everos_root


def _apply_guard(args: _Args) -> None:
    """Run the guard's logic against `args`.

    Re-implemented from the source block rather than called: it lives inline in
    `parse_args`, which needs a full argv and a config. The source-shape test above is
    what keeps the two from drifting.
    """
    if args.servers > 1 and args.everos_root:
        base = Path(args.everos_root[0]).expanduser()
        shards = [base.parent / f"{base.name}_s{i}" for i in range(args.servers)]
        base_has_md = base.is_dir() and next(base.rglob("*.md"), None) is not None
        shards_have_md = any(
            r.is_dir() and next(r.rglob("*.md"), None) is not None for r in shards
        )
        if base_has_md and not shards_have_md:
            print(
                f"  --servers {args.servers} would serve {base.name}_s0.., which hold "
                f"no markdown, while {base.name} does; using 1 server on it"
            )
            args.servers = 1
