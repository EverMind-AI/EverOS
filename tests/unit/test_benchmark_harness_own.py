"""Contracts for the parts of this harness the reference does not have.

These have no counterpart to diff against -- resume, the server fleet, the OME wait, the
conversation selector, the run manifest -- so nothing about them can be settled by
comparison. That is not a reason to leave them untested: two of the worst bugs so far
were here, and both were silent.

* `_done_keys` keyed resume on the question TEXT. LoCoMo conversation 7 has 191
questions
  and 180 distinct strings, so a resumed run dropped 5 and reported "0 to go".
* `_poll_ome` counted every OME retry as a loss, because a retry writes a NEW run_record
  row rather than updating the old one, so healthy runs looked damaged.

Both changed a reported number without raising, which is the failure mode these tests
exist to catch.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run as run_mod  # noqa: E402
from config import BenchmarkConfig, SearchResult  # noqa: E402


def _cfg_for_guard() -> BenchmarkConfig:
    """A config whose only role is to carry `results_root` for the guard.

    The guard reads it to find `<results>/<run>/` and check for `add.done` markers.
    These tests set `results_root` on the args instead, which wins, so an empty one
    here keeps each test's subject the guard's own logic, not config resolution.
    """
    return BenchmarkConfig.from_toml("locomo").model_copy(update={"results_root": ""})


# ---------------------------------------------------------------------------
# Resume: the keys that decide what gets re-run
# ---------------------------------------------------------------------------


def test_done_keys_are_indices_not_question_text(tmp_path: Path) -> None:
    """Two questions with identical text must count as two done rows, not one.

    Keying on the text collapsed duplicates, so a resumed conversation skipped questions
    it had never answered and still printed "0 to go".
    """
    path = tmp_path / "search_hybrid.jsonl"
    rows = [
        {"index": 0, "question": "Same question?"},
        {"index": 1, "question": "Same question?"},
        {"index": 2, "question": "Different"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    keys = run_mod._done_keys(path)
    assert keys == {"0", "1", "2"}, f"expected one key per row, got {keys}"


def test_done_keys_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert run_mod._done_keys(tmp_path / "absent.jsonl") == set()


def test_done_keys_tolerates_a_truncated_last_line(tmp_path: Path) -> None:
    """A run killed mid-write leaves a partial line; the rest must still be usable."""
    path = tmp_path / "search_hybrid.jsonl"
    path.write_text(
        json.dumps({"index": 0, "question": "a"}) + "\n" + '{"index": 1, "quest',
        encoding="utf-8",
    )
    assert "0" in run_mod._done_keys(path)


def test_finalize_jsonl_sorts_by_index_and_rewrites(tmp_path: Path) -> None:
    """Workers append out of order; the artifact must end up in question order."""
    path = tmp_path / "search_hybrid.jsonl"
    out_of_order = [2, 0, 1]
    with open(path, "w", encoding="utf-8") as f:
        for i in out_of_order:
            f.write(
                SearchResult(
                    index=i,
                    question=f"q{i}",
                    golden_answer="g",
                    category=1,
                    evidence=[],
                    episodes=[],
                    profiles=[],
                    search_time_s=0.0,
                    method="hybrid",
                ).model_dump_json()
                + "\n"
            )

    results = run_mod._finalize_jsonl(path, SearchResult)
    assert [r.index for r in results] == [0, 1, 2]
    on_disk = [
        json.loads(x)["index"] for x in path.read_text(encoding="utf-8").splitlines()
    ]
    assert on_disk == [0, 1, 2], (
        "the file itself must be reordered, not just the return"
    )


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    made = [
        SearchResult(
            index=i,
            question=f"q{i}",
            golden_answer="g",
            category=i,
            evidence=[],
            episodes=[],
            profiles=[],
            search_time_s=0.1,
            method="hybrid",
        )
        for i in range(3)
    ]
    for r in made:
        run_mod._append_jsonl(path, r)
    back = run_mod._read_jsonl(path, SearchResult)
    assert [r.index for r in back] == [0, 1, 2]
    assert [r.question for r in back] == ["q0", "q1", "q2"]


# ---------------------------------------------------------------------------
# OME wait: distinguishing a retry from a loss
# ---------------------------------------------------------------------------


def _ome_db(path: Path, rows: list[tuple[str, str, int, int]]) -> None:
    """Build a minimal ome.db with the columns _poll_ome actually reads."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE run_record (event_id TEXT, strategy_name TEXT, status TEXT, "
        "attempt INTEGER, max_retries_snapshot INTEGER, started_at TEXT, "
        "target_path TEXT)"
    )
    for event_id, status, attempt, max_retries in rows:
        conn.execute(
            "INSERT INTO run_record VALUES (?,?,?,?,?,?,?)",
            (
                event_id,
                "episode",
                status,
                attempt,
                max_retries,
                "2026-01-01T00:00:00",
                "/x/default/caroline_conv0/episodes/a.md",
            ),
        )
    conn.commit()
    conn.close()


_SINCE = "2020-01-01T00:00:00"
_FILTER = "target_path LIKE ?"
_PARAMS = ("%caroline_conv0%",)


def test_poll_ome_does_not_count_a_retry_that_later_succeeded(tmp_path: Path) -> None:
    """A retry writes a NEW row for the event, so an earlier failure is not a loss.

    Counting rows with status='failed' read 25 losses on LoCoMo where all 22 affected
    events had succeeded on retry and the store was complete.
    """
    db = tmp_path / "ome.db"
    _ome_db(
        db, [("e1", "failed", 0, 2), ("e1", "failed", 1, 2), ("e1", "success", 2, 2)]
    )
    total, _pending, failed = run_mod._poll_ome(db, _SINCE, _FILTER, _PARAMS)
    assert failed == 0, "a retried-then-succeeded event is not a loss"
    assert total == 3, "every attempt row still counts toward the total"


def test_poll_ome_counts_an_exhausted_event_once(tmp_path: Path) -> None:
    """Only an event with its retry budget spent and no success is genuinely lost."""
    db = tmp_path / "ome.db"
    _ome_db(
        db, [("e2", "failed", 0, 2), ("e2", "failed", 1, 2), ("e2", "failed", 2, 2)]
    )
    _total, _pending, failed = run_mod._poll_ome(db, _SINCE, _FILTER, _PARAMS)
    assert failed == 1, "three attempts on one event is one loss, not three"


def test_poll_ome_ignores_a_failure_whose_retry_is_still_queued(tmp_path: Path) -> None:
    """A retry lands up to 40 min later; a fresh attempt=0 failure is not yet lost."""
    db = tmp_path / "ome.db"
    _ome_db(db, [("e3", "failed", 0, 2)])
    _total, _pending, failed = run_mod._poll_ome(db, _SINCE, _FILTER, _PARAMS)
    assert failed == 0, "a failure with retries left must not be reported as lost"


def test_poll_ome_counts_running_as_pending(tmp_path: Path) -> None:
    db = tmp_path / "ome.db"
    _ome_db(db, [("e4", "running", 0, 2), ("e5", "success", 0, 2)])
    total, pending, failed = run_mod._poll_ome(db, _SINCE, _FILTER, _PARAMS)
    assert (total, pending, failed) == (2, 1, 0)


def test_poll_ome_on_a_missing_db_is_not_a_failure(tmp_path: Path) -> None:
    assert run_mod._poll_ome(tmp_path / "absent.db", _SINCE, _FILTER, _PARAMS) == (
        0,
        0,
        0,
    )


# ---------------------------------------------------------------------------
# Conversation selection
# ---------------------------------------------------------------------------


CONV_SPECS = [
    (["0"], 10, [0]),
    (["0", "3"], 10, [0, 3]),
    (["0-4"], 10, [0, 1, 2, 3, 4]),
    (["all"], 5, [0, 1, 2, 3, 4]),
    (["2-2"], 10, [2]),
    (["0", "0-1"], 10, [0, 1]),
]


@pytest.mark.parametrize("spec,total,want", CONV_SPECS)
def test_parse_conv_spec(spec: list[str], total: int, want: list[int]) -> None:
    """`--conv` scopes a partial run; an off-by-one changes the denominator."""
    assert run_mod._parse_conv_spec(spec, total) == want


def test_parse_conv_spec_rejects_an_index_the_dataset_lacks() -> None:
    """An index past the end must be refused, not passed on.

    Reaching the loader it either raises deep inside a stage or, worse, silently shrinks
    the denominator the accuracy is computed over.
    """
    with pytest.raises(SystemExit, match="does not have"):
        run_mod._parse_conv_spec(["99"], 10)
    with pytest.raises(SystemExit):
        run_mod._parse_conv_spec(["0-99"], 10)
    with pytest.raises(SystemExit):
        run_mod._parse_conv_spec(["-1"], 10)


def test_parse_conv_spec_default_is_every_conversation() -> None:
    assert run_mod._parse_conv_spec(None, 4) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# The stale-store guard
# ---------------------------------------------------------------------------


def _fake_store(root: Path, md: bool) -> None:
    d = root / "default_app" / "default_project" / "users" / "o" / "episodes"
    d.mkdir(parents=True, exist_ok=True)
    if md:
        (d / "episode-2026-01-01.md").write_text("# memory", encoding="utf-8")


class _Args:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def test_guard_refuses_a_populated_store_with_no_markers(tmp_path: Path) -> None:
    """ADD is not idempotent: re-adding into a populated store duplicates memories."""
    store = tmp_path / "s"
    _fake_store(store, md=True)
    args = _Args(
        everos_root=[str(store)],
        servers=0,
        results_root=str(tmp_path / "res"),
        run_name="r",
        stages=["add"],
        reuse_store=False,
    )
    with pytest.raises(SystemExit) as exc:
        run_mod._guard_stale_store(args, _cfg_for_guard())
    assert exc.value.code == 2


def test_guard_allows_an_empty_store(tmp_path: Path) -> None:
    store = tmp_path / "s"
    _fake_store(store, md=False)
    args = _Args(
        everos_root=[str(store)],
        servers=0,
        results_root=str(tmp_path / "res"),
        run_name="r",
        stages=["add"],
        reuse_store=False,
    )
    run_mod._guard_stale_store(args, _cfg_for_guard())


def test_guard_allows_a_genuine_resume(tmp_path: Path) -> None:
    """A populated store WITH markers is an ordinary resume; ADD skips what is done."""
    store = tmp_path / "s"
    _fake_store(store, md=True)
    res = tmp_path / "res" / "r" / "conv0"
    res.mkdir(parents=True)
    (res / "add.done").write_text("{}", encoding="utf-8")
    args = _Args(
        everos_root=[str(store)],
        servers=0,
        results_root=str(tmp_path / "res"),
        run_name="r",
        stages=["add"],
        reuse_store=False,
    )
    run_mod._guard_stale_store(args, _cfg_for_guard())


def test_guard_honours_reuse_store(tmp_path: Path) -> None:
    store = tmp_path / "s"
    _fake_store(store, md=True)
    args = _Args(
        everos_root=[str(store)],
        servers=0,
        results_root=str(tmp_path / "res"),
        run_name="r",
        stages=["add"],
        reuse_store=True,
    )
    run_mod._guard_stale_store(args, _cfg_for_guard())


# ---------------------------------------------------------------------------
# Failure aggregation
# ---------------------------------------------------------------------------


def test_check_failures_raises_on_a_defect() -> None:
    """A deliberate divergence: an exception here can only be a bug in this harness.

    Every API and model failure is handled inside the item that hit it, so converting
    one into an `[ERROR]` row -- as the reference does -- would turn a broken stage into
    a low accuracy number nobody investigates.
    """
    with pytest.raises(ValueError):
        run_mod._check_failures([1, ValueError("boom"), 3])


def test_check_failures_reports_every_one() -> None:
    with pytest.raises(RuntimeError, match="2 failures"):
        run_mod._check_failures([ValueError("a"), 1, KeyError("b")])


def test_check_failures_passes_a_clean_batch() -> None:
    run_mod._check_failures([1, 2, 3])


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ds", ["locomo", "longmemeval", "subtlememory", "evermembench"]
)
def test_adapter_dispatch_reaches_the_named_benchmark(ds: str) -> None:
    """The wrong adapter would load the wrong data and still produce a number."""
    cfg = BenchmarkConfig.from_toml(ds)
    assert cfg.adapter == ds
    keys = run_mod.qa_meta_keys_for(cfg)
    assert isinstance(keys, tuple)
    if ds in ("subtlememory", "evermembench"):
        assert keys, f"{ds}'s judge needs metadata carried through"


def test_prompt_lookup_prefers_the_adapter_over_the_shared_default() -> None:
    """Each benchmark's own prompt must win; the default is the LoCoMo one."""
    for ds in ("locomo", "longmemeval", "subtlememory", "evermembench"):
        cfg = BenchmarkConfig.from_toml(ds)
        got = run_mod._prompt(cfg, "ANSWER_PROMPT")
        import importlib

        want = importlib.import_module(f"adapters.{ds}").ANSWER_PROMPT
        assert got == want, f"{ds}: _prompt returned another benchmark's prompt"


# ---------------------------------------------------------------------------
# The run manifest
# ---------------------------------------------------------------------------


def test_run_spec_records_what_produced_the_number(tmp_path: Path) -> None:
    """Without the manifest a result cannot be attributed to a configuration."""
    cfg = BenchmarkConfig.from_toml("locomo")
    run_mod._write_run_spec(
        tmp_path, "r", cfg, [0, 1], ["search", "answer", "judge"], benchmark="locomo"
    )
    spec = json.loads((tmp_path / "run_spec.json").read_text(encoding="utf-8"))
    assert spec["run_name"] == "r"
    assert spec["conversations"] == [0, 1]
    assert spec["stages"] == ["search", "answer", "judge"]
    # The graded parameters have to be recoverable from the artifact alone.
    blob = json.dumps(spec)
    for value in (cfg.answer_model, cfg.judge_model, str(cfg.top_k)):
        assert value in blob, f"run_spec does not record {value!r}"


def test_run_spec_names_the_benchmark(tmp_path: Path) -> None:
    """Four benchmarks share one results tree; the manifest says which one this was."""
    cfg = BenchmarkConfig.from_toml("subtlememory")
    run_mod._write_run_spec(
        tmp_path, "r", cfg, [0], ["search"], benchmark="subtlememory"
    )
    blob = (tmp_path / "run_spec.json").read_text(encoding="utf-8")
    assert "subtlememory" in blob


def test_stratified_sample_is_reproducible_and_covers_categories() -> None:
    """The same subset must come back next time, or two runs are not comparable."""
    qa = [{"question": f"q{i}", "category": i % 4} for i in range(200)]
    a = run_mod._stratified_sample(qa, n=20)
    b = run_mod._stratified_sample(qa, n=20)
    assert [q["question"] for q in a] == [q["question"] for q in b], (
        "sampling is unstable"
    )
    assert len(a) == 20
    assert len({q["category"] for q in a}) == 4, "every category should be represented"


# ---------------------------------------------------------------------------
# The cascade wait and the report aggregator
# ---------------------------------------------------------------------------


def _cascade_db(path: Path, rows: list[tuple[str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE md_change_state (md_path TEXT, status TEXT)")
    conn.executemany("INSERT INTO md_change_state VALUES (?,?)", rows)
    conn.commit()
    conn.close()


def test_poll_cascade_counts_pending_and_total(tmp_path: Path) -> None:
    """ADD is only finished once the cascade queue for this conversation has drained.

    Reading the wrong statuses as pending would either return early -- leaving episodes
    unindexed when SEARCH starts -- or never return at all.
    """
    db = tmp_path / "system.db"
    _cascade_db(
        db,
        [
            ("/s/default/caroline_conv0/episodes/a.md", "pending"),
            ("/s/default/caroline_conv0/episodes/b.md", "processing"),
            ("/s/default/caroline_conv0/episodes/c.md", "done"),
            ("/s/default/other_conv9/episodes/d.md", "pending"),
        ],
    )
    total, pending = run_mod._poll_cascade(db, "%caroline_conv0%")
    assert total == 3, "the filter must exclude other conversations"
    assert pending == 2, "pending and processing both count as not yet drained"


def test_poll_cascade_drained_queue_reports_zero_pending(tmp_path: Path) -> None:
    db = tmp_path / "system.db"
    _cascade_db(db, [("/s/default/caroline_conv0/episodes/a.md", "done")])
    total, pending = run_mod._poll_cascade(db, "%caroline_conv0%")
    assert (total, pending) == (1, 0)


def test_poll_cascade_on_a_missing_db(tmp_path: Path) -> None:
    assert run_mod._poll_cascade(tmp_path / "absent.db", "%") == (0, 0)


def test_aggregate_report_writes_both_artifacts(tmp_path: Path) -> None:
    """The two report files are the run's output; neither one wastes the whole run."""
    cfg = BenchmarkConfig.from_toml("locomo")
    # The aggregator looks for artifacts named after the configured method, so the
    # fixture has to use that name rather than a hardcoded one.
    method = cfg.parsed_methods[0]
    conv = tmp_path / "conv0"
    conv.mkdir()
    # A finished conversation, written directly: this test is about the aggregator, and
    # driving the stages here would couple it to their fixtures.
    row = {
        "index": 0,
        "question": "q",
        "golden_answer": "g",
        "generated_answer": "a",
        "category": 1,
        "question_id": "q0",
    }
    (conv / f"search_{method}.jsonl").write_text(
        json.dumps(
            {
                **row,
                "evidence": [],
                "episodes": [],
                "profiles": [],
                "search_time_s": 0.1,
                "method": method,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (conv / f"answer_{method}.jsonl").write_text(
        json.dumps(
            {
                **row,
                "answer_time_s": 0.1,
                "answer_attempts": 1,
                "answer_tokens": 30,
                "answer_prompt_tokens": 25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (conv / f"judge_{method}.jsonl").write_text(
        json.dumps({**row, "is_correct": True, "judgments": [True], "judge_tokens": 5})
        + "\n",
        encoding="utf-8",
    )

    run_mod.aggregate_report(tmp_path, [0], cfg)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.txt").exists()
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    blob = json.dumps(data)
    assert method in blob, f"the {method} summary is missing from the report"
    assert "accuracy" in blob.lower()
    assert data[method]["total"] == 1


def test_aggregate_report_on_an_empty_directory_does_not_invent_a_number(
    tmp_path: Path,
) -> None:
    """No judge rows means no accuracy. Reporting 0% or 100% here would be a lie."""
    cfg = BenchmarkConfig.from_toml("locomo")
    run_mod.aggregate_report(tmp_path, [0], cfg)
    report = tmp_path / "report.json"
    if not report.exists():
        return
    data = json.loads(report.read_text(encoding="utf-8"))
    for method in (data.get("methods") or {}).values():
        assert not method or method.get("total", 0) == 0, (
            f"an empty run reported a total of {method.get('total')}"
        )


# ---------------------------------------------------------------------------
# Port-race classification and the SubtleMemory v5 two-stage helpers. Neither had
# a test naming it, which `audit/coverage.py` reports as "no check at all" -- the
# inventory's whole purpose being to make that visible rather than assumed.
# ---------------------------------------------------------------------------


def test_an_empty_server_log_counts_as_a_lost_bind_race() -> None:
    """The consequential half of the rule, and the counter-intuitive one.

    A child killed before it wrote anything is exactly what losing a port race looks
    like. Reading that as a hard error turns a recoverable collision into a failed run,
    so the absence of evidence has to resolve toward "retry at another port block".
    """
    run = run_mod
    assert run._looks_like_bind_failure("") is True
    assert run._looks_like_bind_failure("(no server log)") is True
    assert run._looks_like_bind_failure("   ") is True


def test_only_a_bind_message_justifies_another_port_block() -> None:
    """A server that died of something else must not be retried into a new port.

    Retrying a real crash at another port block spends the whole retry budget on a
    fault that will recur, and reports the exhaustion rather than the cause.
    """
    run = run_mod
    for marker in run._BIND_FAILURE_MARKERS:
        assert run._looks_like_bind_failure(f"OSError: {marker.upper()}") is True
    assert run._looks_like_bind_failure("Traceback: EngineLockHeldError") is False
    assert run._looks_like_bind_failure("CUDA out of memory") is False


def test_v5_fact_parse_returns_empty_rather_than_guessing() -> None:
    """An unparseable stage-1 reply must not become a routing decision.

    The caller reads ``[]`` as "no clash" and falls back to the official prompt, so a
    parse failure degrades to the reference behaviour instead of inventing facts.
    """
    sm = importlib.import_module("adapters.subtlememory")
    assert sm._parse_v5_facts("") == []
    assert sm._parse_v5_facts("not json at all") == []


def test_v5_facts_survive_a_fenced_reply_and_an_unknown_role() -> None:
    """Two things the model does that the parser has to absorb.

    Models fence their JSON, and they invent role names. An unrecognised role becomes
    ``background`` rather than being dropped, because losing the fact loses more than
    mislabelling it.
    """
    sm = importlib.import_module("adapters.subtlememory")
    facts = sm._parse_v5_facts(
        '```json\n{"facts": [{"fact": "moved to Berlin", "role": "invented",'
        ' "source": "s3"}]}\n```'
    )
    assert [f["fact"] for f in facts] == ["moved to Berlin"]
    assert facts[0]["role"] == "background"
    assert facts[0]["role"] in sm._V5_ROLES


def test_v5_render_is_numbered_and_names_its_source() -> None:
    """Stage 2 cites by index, so the numbering is part of the contract."""
    sm = importlib.import_module("adapters.subtlememory")
    out = sm._render_v5_facts(
        [{"fact": "moved to Berlin", "role": "anchor", "source": "session_3"}]
    )
    assert out == "[1] (anchor) moved to Berlin <- session_3"
    assert sm._render_v5_facts([]) == "(no facts extracted)"


def test_log_tail_degrades_to_a_marker_the_bind_check_understands(
    tmp_path: Path,
) -> None:
    """Its failure value is load-bearing, not cosmetic.

    ``_log_tail`` feeds ``_looks_like_bind_failure``, and its "(no server log)" string
    is one of the two inputs that classify as a lost port race. Returning "" or raising
    instead would turn an unreadable log into a hard failure, and a recoverable
    collision into a failed run -- so the marker is part of the contract between them.
    """
    fleet = run_mod._ServerFleet.__new__(run_mod._ServerFleet)
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"line {i}" for i in range(30)), encoding="utf-8")
    fleet.urls, fleet.logs = ["http://127.0.0.1:9999"], [log]

    tail = fleet._log_tail("http://127.0.0.1:9999", lines=3)
    assert tail.splitlines() == ["line 27", "line 28", "line 29"]

    # An unknown url, and a path that is not there: both take the marker branch.
    assert fleet._log_tail("http://127.0.0.1:1") == "(no server log)"
    fleet.logs = [tmp_path / "absent.log"]
    assert fleet._log_tail("http://127.0.0.1:9999") == "(no server log)"
    assert run_mod._looks_like_bind_failure("(no server log)") is True
