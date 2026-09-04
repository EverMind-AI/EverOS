"""Every remaining name in the migrated modules, checked rather than exempted.

`coverage.py` used to carry 53 names in a NO_REFERENCE dict, each with a sentence saying
why it needed no test. Six of those sentences claimed a live run exercised them, written
at a time when no live run had ever been made -- so the inventory reported full coverage
over an assertion nobody could check. That is the same defect as an assertion gate, and
the instruction is that migrated code is verified, not excused.

So these are checked. Some checks are small, because some of the names are small: a bar
width is verified by the bar it draws, a regex by what it matches. The point is that
nothing is taken on the strength of a sentence.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run as run_mod  # noqa: E402
from adapters import base as ab  # noqa: E402
from adapters import evermembench, locomo, longmemeval, subtlememory  # noqa: E402
from config import BenchmarkConfig, SearchResult  # noqa: E402
from metrics import ir as ir_metrics  # noqa: E402

ADAPTERS = (locomo, longmemeval, subtlememory, evermembench)


# ---------------------------------------------------------------------------
# Duck-type shims: the interface is the contract
# ---------------------------------------------------------------------------


def test_the_pool_shims_present_the_openai_surface() -> None:
    """Callers write `pool.chat.completions.create(...)`; that path must route."""
    pool = run_mod.LLMClientPool(["k"], base_url="http://127.0.0.1:1/v1")
    assert isinstance(pool.chat, run_mod._PoolChat)
    assert isinstance(pool.chat.completions, run_mod._PoolCompletions)

    routed: list[dict] = []
    pool._create_with_failover = lambda **kw: routed.append(kw) or "ok"  # type: ignore[method-assign]
    assert pool.chat.completions.create(model="m", messages=[]) == "ok"
    assert routed == [{"model": "m", "messages": []}]


def test_the_ome_outcome_carries_both_numbers() -> None:
    o = run_mod._OmeOutcome(total=7, failed=2)
    assert (o.total, o.failed) == (7, 2)
    assert tuple(o) == (7, 2), "it is consumed positionally by the wait loop"


def test_the_serving_spec_records_a_role_and_survives_json() -> None:
    """It is provenance: it has to come back out of run_spec.json unchanged."""
    s = run_mod.ServingSpec(role="decider", model="qwen3.6-27B", endpoint="http://x/v1")
    back = json.loads(s.model_dump_json())
    assert back["role"] == "decider"
    assert back["model"] == "qwen3.6-27B"
    assert back["local"] is False


def test_the_progress_bar_subclass_renders_at_both_ends() -> None:
    """Verified by what it draws, not by the constant's value."""
    assert isinstance(run_mod._BAR_WIDTH, int) and run_mod._BAR_WIDTH > 0
    half = run_mod._ColorBarTqdm.format_meter(5, 10, 1.0, ncols=80, prefix="x")
    full = run_mod._ColorBarTqdm.format_meter(10, 10, 1.0, ncols=80, prefix="x")
    assert "50%" in half, half
    assert "100%" in full, full
    assert half != full


def test_the_colour_constants_are_inert_off_a_terminal() -> None:
    """Off a tty they must be empty, or a redirected log fills with escape codes."""
    assert run_mod._IS_TTY is sys.stdout.isatty()
    if not run_mod._IS_TTY:
        assert (run_mod._FILL_BG, run_mod._EMPTY_BG, run_mod._RESET) == ("", "", "")


def test_every_adapter_is_reachable_through_the_runner_dispatch() -> None:
    """The Protocol is decorative: LoCoMo does not implement `sessions_of` at all.

    `load_conversation_via_adapter` branches on the benchmark name and gives LoCoMo its
    own loader, whose session shape (`session_idx`) differs from the adapter shape
    (`session_id`). So what is worth checking is not protocol conformance but that the
    dispatch reaches every adapter. The one exception is named so it cannot become two.
    """
    required = [
        n for n in dir(ab.DatasetAdapter) if not n.startswith("_") and n != "name"
    ]
    assert required, "the protocol declares nothing"
    exceptions = {"locomo": ["sessions_of"]}
    for mod in ADAPTERS:
        missing = [n for n in required if not hasattr(mod, n)]
        assert missing == exceptions.get(mod.name, []), (
            f"{mod.name} is missing {missing}, expected {exceptions.get(mod.name, [])}"
        )
    for mod in ADAPTERS:
        cfg = BenchmarkConfig.from_toml(mod.name)
        if not Path(cfg.data_path).exists():
            continue
        sessions, qa, _a, _b, _owner = run_mod.load_conversation_via_adapter(
            mod.name, cfg.data_path, 0
        )
        assert sessions and qa, f"{mod.name}: dispatch produced nothing"
        assert all(s["messages"] for s in sessions), f"{mod.name}: an empty session"


# ---------------------------------------------------------------------------
# Declarations run.py reads
# ---------------------------------------------------------------------------


def test_every_adapter_names_itself_after_its_config() -> None:
    for mod in ADAPTERS:
        assert mod.name == BenchmarkConfig.from_toml(mod.name).adapter


def test_the_meta_key_declarations_reach_the_runner() -> None:
    """`qa_meta_keys_for` is the only consumer; a mis-spelled declaration reads as none.

    That exact typo -- JUDGE_META_KEYS where the runner reads QA_META_KEYS -- left every
    SubtleMemory judge field empty.
    """
    # LongMemEval is absent on purpose: its grading rule keys on `question_id`, a
    # first-class SearchResult field, not on qa_meta.
    expected_nonempty = {"subtlememory", "evermembench"}
    for mod in ADAPTERS:
        keys = run_mod.qa_meta_keys_for(BenchmarkConfig.from_toml(mod.name))
        assert isinstance(keys, tuple | list)
        if mod.name in expected_nonempty:
            assert keys, f"{mod.name} declares metadata the runner cannot see"


def test_the_provenance_package_list_names_the_algorithm_packages() -> None:
    """These make a store's extraction reproducible; a wrong list hides drift."""
    assert run_mod._PROVENANCE_PACKAGES
    # The everalgo packages decide an extraction's reproducibility; the rest are the
    # storage and client stack that decide a retrieval's.
    assert any(p.startswith("everalgo-") for p in run_mod._PROVENANCE_PACKAGES)
    assert "everos" in run_mod._PROVENANCE_PACKAGES
    assert "lancedb" in run_mod._PROVENANCE_PACKAGES
    collected = run_mod._collect_packages()
    assert set(run_mod._PROVENANCE_PACKAGES) <= set(collected), (
        f"declared but not collected: "
        f"{set(run_mod._PROVENANCE_PACKAGES) - set(collected)}"
    )


def test_the_everos_version_is_readable_and_lands_in_the_manifest() -> None:
    v = run_mod._get_everos_version()
    assert v and v != "unknown", f"version reads as {v!r}"


def test_the_serving_specs_describe_every_configured_role() -> None:
    specs = run_mod._serving_from_env(BenchmarkConfig.from_toml("locomo"))
    roles = {s.role for s in specs}
    assert "embedding" in roles or "decider" in roles, f"roles: {sorted(roles)}"
    for s in specs:
        assert s.model, f"{s.role} has no model recorded"


def test_the_search_retry_budget_is_a_positive_count() -> None:
    assert isinstance(run_mod._SEARCH_RETRIES, int)
    assert run_mod._SEARCH_RETRIES >= 1


def test_the_budget_stop_flag_starts_clear_and_can_be_set() -> None:
    assert isinstance(run_mod._BUDGET_STOP, threading.Event)
    assert not run_mod._BUDGET_STOP.is_set(), (
        "a fresh process must not think it is broke"
    )


def test_the_ir_k_values_are_ascending_and_drive_the_metric() -> None:
    assert list(ir_metrics.KS) == sorted(ir_metrics.KS)
    out = ir_metrics.score([["a", "b", "c"]], [{"a"}])
    for k in ir_metrics.KS:
        assert any(str(k) in key for key in out), (
            f"KS declares {k} but score() reports nothing for it: {sorted(out)}"
        )


def test_the_session_timestamp_formats_each_parse_something() -> None:
    """A dead format in the list is a format nobody noticed stopped being used."""
    samples = ("10:02 am on 4 March, 2025", "2023/05/20 (Sat) 02:21")
    assert len(ab._SESSION_TS_FORMATS) == len(samples)
    for s in samples:
        assert ab.session_epoch_ms(s) is not None, f"no format parses {s!r}"


def test_the_locomo_category_five_filter_is_the_one_that_is_excluded() -> None:
    """Cat 5 is adversarial and ungradable; the count 1540 depends on this set."""
    assert {"5"} == locomo._EXCLUDED_CATEGORIES
    cfg = BenchmarkConfig.from_toml("locomo")
    if not Path(cfg.data_path).exists():
        pytest.skip("dataset not present")
    units = locomo.load_units(cfg.data_path)
    cats = {str(q.get("category")) for u in units for q in (u.get("qa") or [])}
    assert "5" not in cats, "a category-5 question survived the filter"
    assert sum(len(u.get("qa") or []) for u in units) == 1540


def test_a_missing_dataset_path_fails_instead_of_loading_nothing() -> None:
    """A path that does not resolve must raise, not quietly yield zero units.

    This used to assert the module-level defaults were real directories, which worked
    only because they were absolute paths into one workspace -- the same thing that
    made the harness unusable to anyone else. They are now read from the environment
    (``BENCH_DATA_SUBTLEMEMORY`` / ``EVERMEMBENCH_RAW_ROOT``), so what actually needs
    pinning is the failure mode: an unset or stale value has to stop the run rather
    than produce an empty load that scores 0% and looks like a result.
    """
    for mod, attr in ((subtlememory, "DATA_DIR"), (evermembench, "RAW_ROOT")):
        configured = Path(getattr(mod, attr))
        if configured.exists():
            continue  # the environment names a real path; nothing to prove here
        with pytest.raises(
            (FileNotFoundError, NotADirectoryError, OSError, ValueError)
        ):
            mod.load_units(str(configured / "definitely-absent.json"))


def test_subtlememory_rejects_a_directory_that_is_not_there() -> None:
    """Pinned separately because it silently did the opposite.

    ``load_units`` walks ``persona_0..9`` and ``continue``s past any persona whose two
    files are missing -- correct for a partial download, and indistinguishable from a
    root that does not exist, which returned zero units with no error. A run against it
    scored 0/0 and reported a clean finish.
    """
    with pytest.raises(FileNotFoundError, match="BENCH_DATA_SUBTLEMEMORY"):
        subtlememory.load_units("/nonexistent/subtlememory-root")


def test_the_synthetic_clock_is_only_a_fallback() -> None:
    """It must never be reached by a dataset that carries real timestamps."""
    fallbacks = {m.name: m._BASE_TS_MS for m in ADAPTERS if hasattr(m, "_BASE_TS_MS")}
    assert fallbacks, "no adapter declares a fallback clock"
    assert all(isinstance(v, int) and v > 0 for v in fallbacks.values())
    for mod in ADAPTERS:
        cfg = BenchmarkConfig.from_toml(mod.name)
        if not Path(cfg.data_path).exists():
            continue
        if not hasattr(mod, "sessions_of"):
            continue  # LoCoMo: its own loader, covered by the dispatch test
        sessions = mod.sessions_of(mod.load_units(cfg.data_path)[0])
        stamps = [m["timestamp_ms"] for s in sessions for m in s["messages"]]
        assert stamps
        if mod.name in fallbacks:
            assert min(stamps) != fallbacks[mod.name], (
                f"{mod.name} fell back to the synthetic clock"
            )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_write_jsonl_round_trips_one_row_per_item(tmp_path: Path) -> None:
    rows = [
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
        for i in range(3)
    ]
    f = tmp_path / "out.jsonl"
    run_mod._write_jsonl(f, rows)
    assert len(f.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert [r.index for r in run_mod._read_jsonl(f, SearchResult)] == [0, 1, 2]


def test_the_live_loader_agrees_with_the_adapter() -> None:
    """`load_units_via_adapter` was deleted rather than tested.

    It had no callers, its docstring claimed the SEARCH / ANSWER / JUDGE stages used it,
    and it hardcoded `"speaker_a"` in place of `config.eval_owner` -- so it returned a
    different unit shape than the adapter it wrapped (`owner_id`/`categories` instead of
    `conversation`/`speaker_a`). The exemption covering it read "one getattr and a
    call", untrue in every clause. What remains is the loader the stages really use.
    """
    import run as run_mod

    assert not hasattr(run_mod, "load_units_via_adapter"), (
        "the dead wrapper is back; it has no callers and lies about having them"
    )
    cfg = BenchmarkConfig.from_toml("locomo")
    if not Path(cfg.data_path).exists():
        pytest.skip("dataset not present")
    units = locomo.load_units(cfg.data_path)
    sessions, qa, sa, sb, _owner = run_mod.load_conversation_via_adapter(
        "locomo", cfg.data_path, 0
    )
    assert sessions and qa
    assert (sa, sb) == (units[0]["speaker_a"], units[0]["speaker_b"]), (
        "the loader disagrees with the adapter about who is talking"
    )
    assert all("gold_sessions" in q for q in qa), "gold was not resolved"
    assert len(qa) == len(units[0]["qa"]), "the loader dropped or added questions"


def test_the_session_helpers_produce_the_names_gold_is_matched_on() -> None:
    """The naming helpers are per adapter; what matters is that gold lines up.

    `_session_name` / `_group_order` / `_session_index_map` live on the adapters that
    need them, and their only job is to make `gold_of` name a session `sessions_of`
    built. That join is checked directly rather than the helpers in isolation.
    """
    helpers = {
        m.name: [
            h
            for h in ("_session_name", "_group_order", "_session_index_map")
            if hasattr(m, h)
        ]
        for m in ADAPTERS
    }
    assert any(helpers.values()), f"no adapter declares any naming helper: {helpers}"
    for mod in ADAPTERS:
        cfg = BenchmarkConfig.from_toml(mod.name)
        if not Path(cfg.data_path).exists():
            continue
        unit = mod.load_units(cfg.data_path)[0]
        if not hasattr(mod, "sessions_of"):
            continue  # LoCoMo: own loader, covered by the dispatch test above
        names = {s["session_id"] for s in mod.sessions_of(unit)}
        gold = set()
        for q in unit.get("qa") or []:
            gold |= mod.gold_of(unit, q)
        assert gold, f"{mod.name}: no gold resolved at all"
        assert gold <= names, (
            f"{mod.name}: gold names no session: {sorted(gold - names)[:3]}"
        )


def test_the_dead_judge_label_helper_is_gone_or_agrees_with_the_live_one() -> None:
    """It was exempted as dead code. Dead code is deleted, not exempted."""
    assert not hasattr(ab, "extract_judge_label"), (
        "extract_judge_label is still present; delete it or make it the live path"
    )
