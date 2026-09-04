"""Failure modes a real run hits that no reference comparison can find.

These are not parity defects -- the reference has no equivalent of any of them. They are
ways this harness breaks, silently, on the second attempt after something went wrong:
a process killed mid-write, a descending `--conv` range, a guard looking at a directory
the fleet does not use, a count reconstructed from a rounded percentage. Every one was
reachable with the code as written and is exercised here.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run as run_mod  # noqa: E402
from config import AnswerResult, JudgeResult, SearchResult  # noqa: E402


def _rows(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            index=i,
            question=f"q{i}",
            golden_answer="a",
            category=1,
            episodes=[],
            profiles=[],
            evidence=[],
            method="x",
            search_time_s=0.0,
        )
        for i in range(n)
    ]


def _torn(path: Path, rows: list[Any]) -> None:
    """As a process killed mid-write leaves it: whole lines plus a partial one."""
    body = "\n".join(r.model_dump_json() for r in rows)
    path.write_text(body + "\n" + rows[0].model_dump_json()[:40], encoding="utf-8")


def _present(path: Path) -> bool:
    """``Path.exists()`` that answers False instead of raising on EACCES.

    ``pathlib`` only swallows ENOENT / ENOTDIR / EBADF / ELOOP, so a probe for a
    path under a directory the current user cannot stat re-raises. ``/root`` is
    mode 0700 and CI does not run as root, so the private-shard guard below
    raised ``PermissionError`` there and turned its skip into a failure. For a
    guard asking "is this store on this machine", unreadable is not present.
    """
    try:
        return path.exists()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# A torn final line must not lock the stage
# ---------------------------------------------------------------------------


def test_a_torn_final_line_does_not_lock_the_stage(tmp_path: Path) -> None:
    """`_done_keys` was written to tolerate this; every other reader was not.

    So the interrupted run counted its work as done, the retry did the remaining
    questions and appended them, and then died in `_finalize_jsonl` -- every time, for
    ever. The stage could never complete again without deleting the file.
    """
    f = tmp_path / "search_x.jsonl"
    _torn(f, _rows(3))

    assert sorted(run_mod._done_keys(f)) == ["0", "1", "2"], (
        "the tolerant reader changed; this test's premise is gone"
    )
    out = run_mod._finalize_jsonl(f, SearchResult)
    assert len(out) == 3
    # And the file is left clean, so the next read has nothing to trip over.
    assert len(f.read_text(encoding="utf-8").strip().splitlines()) == 3
    run_mod._finalize_jsonl(f, SearchResult)


def test_a_torn_upstream_file_does_not_stop_the_next_stage(tmp_path: Path) -> None:
    """The answer stage reads the search file, and the judge stage the answer file."""
    search = tmp_path / "search_x.jsonl"
    _torn(search, _rows(3))
    assert len(run_mod._read_jsonl(search, SearchResult, strict=False)) == 3

    answers = [
        AnswerResult(
            index=i,
            question=f"q{i}",
            golden_answer="a",
            category=1,
            generated_answer="b",
            answer_time_s=0.0,
            answer_attempts=1,
            answer_tokens=1,
        )
        for i in range(3)
    ]
    ans = tmp_path / "answer_x.jsonl"
    _torn(ans, answers)
    assert len(run_mod._read_jsonl(ans, AnswerResult, strict=False)) == 3


def test_a_strict_read_still_raises(tmp_path: Path) -> None:
    """The tolerance is opt-in: a caller that wants the error still gets it."""
    f = tmp_path / "search_x.jsonl"
    _torn(f, _rows(2))
    with pytest.raises(ValueError):
        run_mod._read_jsonl(f, SearchResult)


def test_a_torn_judge_file_does_not_stop_the_report(tmp_path: Path) -> None:
    conv = tmp_path / "conv0"
    conv.mkdir()
    rows = [
        JudgeResult(
            index=i,
            question=f"q{i}",
            golden_answer="a",
            generated_answer="a",
            category=1,
            is_correct=True,
            judgments=[True],
            judge_tokens=1,
        )
        for i in range(3)
    ]
    _torn(conv / "judge_hybrid.jsonl", rows)
    from config import BenchmarkConfig

    s = run_mod._collect_method_summary(
        "hybrid", tmp_path, [0], BenchmarkConfig.from_toml("locomo")
    )
    assert s is not None
    assert s["total"] == 3


# ---------------------------------------------------------------------------
# A `--conv` spec that selects nothing is a mistake, not a run
# ---------------------------------------------------------------------------


def test_a_descending_range_is_refused() -> None:
    """It expanded to an empty list, and the run then finished with an empty report and
    exit 0 -- indistinguishable from a completed run.
    """
    with pytest.raises(SystemExit, match="selects no conversations"):
        run_mod._parse_conv_spec(["5-2"], 10)


def test_an_out_of_range_index_is_still_refused() -> None:
    with pytest.raises(SystemExit):
        run_mod._parse_conv_spec(["99"], 10)


def test_a_valid_spec_still_expands() -> None:
    assert run_mod._parse_conv_spec(["3", "0-2"], 10) == [3, 0, 1, 2]
    assert run_mod._parse_conv_spec(None, 4) == [0, 1, 2, 3]
    assert run_mod._parse_conv_spec(["all"], 3) == [0, 1, 2]


# ---------------------------------------------------------------------------
# The stale-store guard has to look where the fleet actually writes
# ---------------------------------------------------------------------------


def _populated_store() -> Path:
    store = Path(tempfile.mkdtemp()) / "store"
    (store / "default_app").mkdir(parents=True)
    (store / "default_app" / "a.md").write_text("x", encoding="utf-8")
    return store


@pytest.mark.parametrize("servers", [0, 1])
def test_the_guard_refuses_a_populated_store_the_fleet_would_use(servers: int) -> None:
    """One server uses the base root itself; only a fleet appends `_s<i>`.

    Deriving the suffix unconditionally made the guard inspect directories that do not
    exist, so it passed on a populated store whenever `--servers 1` -- and `--servers`
    is clamped to the conversation count, so resuming a single conversation switched the
    guard off exactly when it mattered.
    """
    store = _populated_store()
    args = type(
        "A",
        (),
        {
            "everos_root": [str(store)],
            "servers": servers,
            "results_root": tempfile.mkdtemp(),
            "run_name": "r",
            "reuse_store": False,
            "stages": ["add"],
        },
    )()
    with pytest.raises(SystemExit):
        run_mod._guard_stale_store(
            args,
            run_mod.BenchmarkConfig.from_toml("locomo").model_copy(
                update={"results_root": ""}
            ),
        )


def test_the_guard_looks_at_the_shard_roots_for_a_real_fleet() -> None:
    """With two servers the fleet uses `_s0`/`_s1`, not the bare root."""
    store = _populated_store()
    args = type(
        "A",
        (),
        {
            "everos_root": [str(store)],
            "servers": 2,
            "results_root": tempfile.mkdtemp(),
            "run_name": "r",
            "reuse_store": False,
            "stages": ["add"],
        },
    )()
    run_mod._guard_stale_store(
        args,
        run_mod.BenchmarkConfig.from_toml("locomo").model_copy(
            update={"results_root": ""}
        ),
    )


# ---------------------------------------------------------------------------
# A count must not be reconstructed from a rounded rate
# ---------------------------------------------------------------------------


def test_the_unanimity_count_is_carried_not_recomputed(tmp_path: Path) -> None:
    """`unanimous_rate` is rounded to three places.

    Multiplying it back by the row count prints the wrong number for 756 of the
    (agree, total) pairs with total up to 60: 4 of 7 reads as 3, and 1 of 3 as 0.
    """
    conv = tmp_path / "conv0"
    conv.mkdir()
    # Seven rows, four of which are unanimous across three judge runs.
    rows = []
    for i in range(7):
        unanimous = i < 4
        judgments = [True, True, True] if unanimous else [True, False, True]
        rows.append(
            JudgeResult(
                index=i,
                question=f"q{i}",
                golden_answer="a",
                generated_answer="a",
                category=1,
                is_correct=True,
                judgments=judgments,
                judge_tokens=1,
            )
        )
    (conv / "judge_hybrid.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in rows), encoding="utf-8"
    )
    from config import BenchmarkConfig

    s = run_mod._collect_method_summary(
        "hybrid", tmp_path, [0], BenchmarkConfig.from_toml("locomo")
    )
    assert s is not None
    assert s["judge"]["unanimous"] == 4, "the count itself is not reported"
    assert int(s["judge"]["unanimous_rate"] * s["judge"]["count"]) == 3, (
        "the rounding hazard is gone, so this test no longer proves anything"
    )


def test_the_rounding_hazard_is_real_across_small_totals() -> None:
    """Recorded so the reason the count is carried does not get optimised away."""
    wrong = [
        (agree, total)
        for total in range(1, 61)
        for agree in range(total + 1)
        if int(round(agree / total, 3) * total) != agree
    ]
    assert len(wrong) == 756, f"{len(wrong)} pairs reconstruct wrongly"
    assert (4, 7) in wrong
    assert (1, 3) in wrong


def test_the_report_prints_the_carried_count(tmp_path: Path) -> None:
    """Carrying the count is only half of it -- the writer has to use it.

    Asserting the summary dict alone left `report.txt` free to keep multiplying the
    rounded rate, which is where the wrong number was actually shown.
    """
    from config import BenchmarkConfig

    summary = {
        "method": "hybrid",
        "total": 7,
        "correct": 7,
        "accuracy": 1.0,
        "mean_accuracy": 1.0,
        "max_accuracy": 1.0,
        "per_run_accuracies": [1.0],
        "category_stats": {},
        "per_conversation": {},
        "scored_conversations": [0],
        "requested_conversations": [0],
        "judge_excluded": 0,
        "search": {
            "count": 7,
            "avg_latency_s": 0,
            "p50_latency_s": 0,
            "max_latency_s": 0,
        },
        "answer": {
            "count": 7,
            "avg_latency_s": 0,
            "total_tokens": 0,
            "avg_prompt_tokens": 0,
            "retries": 0,
        },
        # Four of seven unanimous: the rate rounds to 0.571, and 0.571 * 7 floors to 3.
        "judge": {
            "count": 7,
            "unanimous": 4,
            "unanimous_rate": round(4 / 7, 3),
            "total_tokens": 0,
            "judge_runs": 3,
        },
    }
    report = tmp_path / "report.txt"
    run_mod._write_report_txt(
        report,
        {"hybrid": summary},
        [0],
        BenchmarkConfig.from_toml("locomo"),
        {"git_hash": "abc", "stages": ["judge"]},
        "0h 1m",
    )
    body = report.read_text(encoding="utf-8")
    line = next(ln for ln in body.splitlines() if "unanimous" in ln)
    # The line carries a percentage, so the discriminating value is the percentage
    # itself: four of seven is 57.1%, while multiplying the rounded rate back gives
    # three of seven and prints 42.9%.
    assert "57.1%" in line, f"the reported unanimity is not 4 of 7: {line!r}"
    assert "42.9%" not in line, f"the rounded rate was multiplied back: {line!r}"


# ---------------------------------------------------------------------------
# Concurrent runs must not send their searches to each other's servers
# ---------------------------------------------------------------------------


def test_port_allocation_cannot_hand_the_same_port_to_two_runs() -> None:
    """Probing then returning the number is a race; this holds the port instead.

    Observed, not theorised: seven runs started at once, two of them picked port 9640,
    one child died at bind, and the loser's searches went to a DIFFERENT dataset's
    server. It answered HTTP 200 with zero episodes for every question and the run
    printed "Done" with a report -- a complete-looking 0.0%. The same collision explains
    an earlier 0.0% whose cause could not be reproduced: three smoke runs had been
    launched together.
    """
    first, held_a = run_mod._ServerFleet._free_port(9900)
    try:
        second, held_b = run_mod._ServerFleet._free_port(9900)
        try:
            assert first != second, (
                f"both allocations returned {first}; a concurrent run would collide"
            )
        finally:
            held_b.close()
    finally:
        held_a.close()


def test_readiness_refuses_a_fleet_whose_child_has_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/health` answering is not proof the answer came from our server.

    It carries no store identity, so a dead child plus somebody else's live server on
    the same port read as ready. The child being alive is what tells the two apart.
    """

    class _Dead:
        returncode = 1

        def poll(self) -> int:
            return 1

        def terminate(self) -> None: ...

        def kill(self) -> None: ...

        def wait(self, timeout: float | None = None) -> int:
            return 1

    root = tmp_path / "store"
    root.mkdir()
    fleet = run_mod._ServerFleet(1, root, first_port=0)
    fleet.urls = ["http://127.0.0.1:9999"]
    fleet.procs = [_Dead()]  # type: ignore[list-item]
    # Something does answer on that port, which is exactly the dangerous case.
    monkeypatch.setattr(
        run_mod.urllib.request if hasattr(run_mod, "urllib") else run_mod,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("unused")),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        fleet._await_ready(timeout_s=1.0)


def test_a_lost_port_race_is_a_distinct_signal_not_a_failure() -> None:
    """It is what lets `start` move to another port block instead of giving up.

    Made a distinct type on purpose: a child that dies at bind is contention, and a run
    that fails on contention wastes the whole launch; a child that dies for any other
    reason is a real fault and must not be retried into oblivion.
    """
    assert issubclass(run_mod._PortRaceLostError, RuntimeError)
    # `start` retries it and `_start_once` is the single attempt it retries.
    import inspect

    start_src = inspect.getsource(run_mod._ServerFleet.start)
    assert "_PortRaceLostError" in start_src
    assert "_start_once" in start_src
    assert hasattr(run_mod._ServerFleet, "_start_once")


def test_the_fleet_moves_to_another_port_block_after_losing_a_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs launched in the same instant both pass through the bind gap.

    Holding the port narrows the window but cannot close it, so the fleet has to survive
    losing. Observed: seven concurrent runs, two picked the same block, one lost, and
    its searches went to another dataset's store and returned zero episodes for every
    question.
    """
    root = tmp_path / "store"
    root.mkdir()
    attempts: list[int] = []

    def _fail_once(self: Any) -> None:
        attempts.append(self._first_port)
        if len(attempts) == 1:
            raise run_mod._PortRaceLostError("child exited at bind")

    monkeypatch.setattr(run_mod._ServerFleet, "_start_once", _fail_once)
    fleet = run_mod._ServerFleet(1, root, first_port=9400)
    monkeypatch.setattr(fleet, "stop", lambda: None)
    fleet.start()
    assert len(attempts) == 2, f"did not retry: {attempts}"
    assert attempts[0] != attempts[1], f"retried the same block: {attempts}"


def test_the_fleet_gives_up_after_enough_lost_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying for ever would hide a machine with no free ports at all."""
    root = tmp_path / "store"
    root.mkdir()
    tried: list[int] = []

    def _always_lose(self: Any) -> None:
        tried.append(self._first_port)
        raise run_mod._PortRaceLostError("child exited at bind")

    monkeypatch.setattr(run_mod._ServerFleet, "_start_once", _always_lose)
    fleet = run_mod._ServerFleet(1, root, first_port=9400)
    monkeypatch.setattr(fleet, "stop", lambda: None)
    with pytest.raises(RuntimeError, match="could not claim a free port block"):
        fleet.start(attempts=3)
    assert len(tried) == 3
    assert len(set(tried)) == 3, f"blocks repeated: {tried}"


# ---------------------------------------------------------------------------
# The store owner is not the answer prompt's speaker
# ---------------------------------------------------------------------------


def test_the_loader_returns_the_owner_rather_than_leaving_it_to_be_guessed() -> None:
    """LongMemEval scored 0.0% on every question because these two were conflated.

    `speakers_of` returns `user_<question_id>` / `assistant_<question_id>` so the answer
    prompt's header matches the reference. The store's owner is `longmemeval_0`.
    `run_conversation` used the speaker as the owner, so every search filtered on an
    owner with zero rows: HTTP 200, zero episodes, 0.1s, and no error anywhere. LoCoMo
    escaped through its own branch; SubtleMemory and EverMemBench because for them the
    two happen to be the same string -- which is exactly why one dataset failing looked
    inexplicable.
    """
    from config import BenchmarkConfig

    expected_differs = {"locomo", "longmemeval"}
    for name in ("locomo", "longmemeval", "subtlememory", "evermembench"):
        cfg = BenchmarkConfig.from_toml(name)
        if not Path(cfg.data_path).exists():
            continue
        out = run_mod.load_conversation_via_adapter(name, cfg.data_path, 0)
        assert len(out) == 5, f"{name}: loader returned {len(out)} values, not 5"
        _s, _q, spk_a, _spk_b, owner = out
        assert owner, f"{name}: no owner returned"
        ad = importlib.import_module(f"adapters.{name}")
        if hasattr(ad, "owner_of"):
            units = ad.load_units(cfg.data_path)
            assert owner == (
                f"{spk_a.lower()}_conv0"
                if name == "locomo"
                else ad.owner_of(units[0], cfg.eval_owner)
            ), f"{name}: loader owner {owner!r} disagrees with the adapter"
        if name in expected_differs:
            assert owner != spk_a, (
                f"{name}: owner and speaker are the same, so this cannot detect "
                f"the conflation it exists for"
            )


def test_the_owner_the_search_uses_has_rows_in_the_store() -> None:
    """The check that would have caught it: does anything in the store answer to it?

    Pinning the string alone would not have: both `user_e47becba` and
    `longmemeval_0` look like plausible owners. Only the store knows which exists.
    """
    import lancedb
    from config import BenchmarkConfig

    shard = Path("/root/v3_longmemeval/store_s0")
    if not _present(shard / ".index" / "lancedb"):
        pytest.skip("LongMemEval shard not present")
    cfg = BenchmarkConfig.from_toml("longmemeval")
    if not Path(cfg.data_path).exists():
        pytest.skip("dataset not present")
    _s, _q, spk_a, _b, owner = run_mod.load_conversation_via_adapter(
        "longmemeval", cfg.data_path, 0
    )
    table = lancedb.connect(str(shard / ".index" / "lancedb")).open_table("episode")
    assert table.count_rows(filter=f"owner_id = '{owner}'") > 0, (
        f"the owner the search would use, {owner!r}, has no rows in the store"
    )
    assert table.count_rows(filter=f"owner_id = '{spk_a}'") == 0, (
        f"{spk_a!r} unexpectedly has rows; this test's premise is gone"
    )
