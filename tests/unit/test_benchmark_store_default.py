"""The store defaults to living with the results it produced.

Before this, `--everos-root` was mandatory and launch.sh answered it with
a directory outside the results tree "for easy cleanup". Every consequence was bad:

* The store sat on a different filesystem from its results. `/dev/vdb` was at 84%,
  and the `[Errno 28] No space left on device` that killed a
  evermembench run landed there -- while the 130T VEPFS holding the results was
  nowhere near full.
* The only record of which store produced which numbers was a path inside
  `run_spec.json`. When a store was deleted, the results it explained silently
  became unreproducible; the tombstone in
  `results/repo/evermembench/STORE_DELETED.md` had to be written by hand.
* Nothing stopped a later run from pointing a different store at the same results
  directory.

So ADD now defaults to `<results>/<run>/store`, and a stage-only run given no root
reuses that same location. An explicit root still wins -- which is what lets a run
score an existing store it did not build, and
the retrieval-policy arms both score against canonical stores they did not build.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

SRC = (_BENCH / "run.py").read_text(encoding="utf-8")


def test_add_defaults_the_store_into_the_results_dir() -> None:
    """ADD with no --everos-root must not fall back to a path outside the results."""
    block = SRC[SRC.index("_default_store = _res_root") :][:900]
    assert '_default_store = _res_root / "store"' in block
    assert 'if "add" in args.stages' in block


def test_stage_only_run_reuses_that_same_location() -> None:
    """Re-running answer/judge should find the store ADD left, without being told."""
    block = SRC[SRC.index("_default_store = _res_root") :][:1200]
    assert "elif _default_store.exists()" in block


def test_a_stage_only_run_with_no_store_anywhere_is_an_error() -> None:
    """Not a silent empty run.

    A missing store used to surface as every search returning nothing, which the
    harness reports as a finished run with 0.0% -- indistinguishable from a real
    result. A retrieval run did exactly that.
    """
    block = SRC[SRC.index("_default_store = _res_root") :][:1600]
    assert "p.error(" in block
    assert "--everos-root is required" in block


def test_an_explicit_root_still_wins() -> None:
    """Canonical-store runs depend on it.

    A run that scores an existing store passes its path in, rather than having one
    derived for it -- which is the whole point of the flag staying available.
    """
    block = SRC[SRC.index("_default_store = _res_root") :][:1200]
    assert "if not args.everos_root" in block
