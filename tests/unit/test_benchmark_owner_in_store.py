"""A store that does not hold a conversation's owner must stop the run.

The failure this prevents leaves no trace anywhere. A search against a store without the
owner returns an empty episode list with HTTP 200; every question is then answered from
nothing, graded wrong, and the run finishes with `exit 0` and a report. The number is
built entirely from absent evidence and nothing in the output says so.

Reaching that state takes one plausible mistake. `--everos-root` accepts a single path
and replicates it across every server (`_roots * len(_urls)`), so a benchmark whose
conversations live in separate shards will read all of them from the first shard.
Measured on EverMemBench, whose five topics live in five stores: pointed at one of them,
conv0 retrieved 10/10 and conv1 retrieved 0/10, and the run reported 40.0% rather than
failing.

The check reads markdown rather than issuing a search: markdown is the store's source of
truth, and an empty search result is the symptom under diagnosis, not a usable test.
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


def _store(tmp: Path, owners: dict[str, bool]) -> Path:
    """A store holding `owners`; the bool says whether the owner has any markdown."""
    users = tmp / "default_app" / "default_project" / "users"
    for name, populated in owners.items():
        d = users / name
        d.mkdir(parents=True, exist_ok=True)
        if populated:
            (d / "episodes").mkdir(exist_ok=True)
            (d / "episodes" / "e1.md").write_text("# memory", encoding="utf-8")
    return tmp


def test_a_populated_owner_passes(tmp_path: Path) -> None:
    """The normal case, and it must not cost a search."""
    root = _store(tmp_path, {"01": True, "02": True})
    run._assert_owner_in_store(str(root), "01", "default", "default", 0)


def test_a_missing_owner_stops_the_run(tmp_path: Path) -> None:
    """The exact shape of the mistake: right store, wrong shard."""
    root = _store(tmp_path, {"01": True})
    with pytest.raises(SystemExit) as e:
        run._assert_owner_in_store(str(root), "02", "default", "default", 1)
    assert "no markdown for owner '02'" in str(e.value)


def test_the_message_names_what_the_store_does_hold(tmp_path: Path) -> None:
    """Naming the occupants is what turns the error into a diagnosis.

    "Owner 02 is missing" leaves the reader guessing whether the store is empty, the
    wrong one, or partitioned differently. "It holds: 01" says which shard they reached.
    """
    root = _store(tmp_path, {"01": True, "03": True})
    with pytest.raises(SystemExit) as e:
        run._assert_owner_in_store(str(root), "02", "default", "default", 1)
    msg = str(e.value)
    assert "It holds: 01, 03" in msg
    # Substring chosen to sit on one line of the message: the sentence wraps for
    # line length, so matching across the break would break on any rewrap.
    assert "--everos-root per shard" in msg


def test_an_owner_directory_with_no_markdown_is_missing(tmp_path: Path) -> None:
    """An empty owner directory is the same failure as no directory.

    A shard created but never ingested has the directory and none of the content, and a
    search against it returns exactly as much as against a shard that lacks it entirely.
    """
    root = _store(tmp_path, {"01": True, "02": False})
    with pytest.raises(SystemExit):
        run._assert_owner_in_store(str(root), "02", "default", "default", 1)


def test_an_absent_users_directory_is_not_evidence(tmp_path: Path) -> None:
    """Do not block a fresh store, or a layout that partitions differently.

    ADD has not written anything yet on a store it is about to build, and other
    deployments key the path differently. Absence of the directory this check knows how
    to read says nothing about the owner, so it must not be read as a verdict.
    """
    run._assert_owner_in_store(str(tmp_path), "01", "default", "default", 0)


def test_the_check_runs_before_the_first_question() -> None:
    """Placed at the top of SEARCH, not inside the per-question loop.

    Later would mean paying for part of the run first, and per-question would mean the
    same message once per question.
    """
    src = (_BENCH / "run.py").read_text(encoding="utf-8")
    block = src[src.index('if "search" in stages:') :][:600]
    assert "_assert_owner_in_store(" in block
    assert block.index("_assert_owner_in_store(") < block.index('_stage("search"')
